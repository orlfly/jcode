# Jcode 记忆存储策略优化方向分析（结合会话历史）

采集时间: 2026-08-28 18:32 CST
数据来源:
- `/home/cc/.jcode/logs/memory-events-*.jsonl`（8 天，含每日事件流）
- `/home/cc/.jcode/logs/memory/client|server-runtime-memory-*.jsonl`
- `/home/cc/.jcode/memory/backend-sqlite/gvec.sqlite`（192 节点 / 224 边）
- `/home/cc/.jcode/prompt-history.jsonl`（538 条，201 个会话）

---

## 1. 核心数据信号（会话历史视角）

### 1.1 召回管线投入产出严重失衡
| 指标 | 数值 | 含义 |
| --- | --- | --- |
| `embedding_started` | 547 次 | 每次用户回合都跑嵌入 |
| `candidate_filter` | 492 次 | 候选过滤 |
| `sidecar_started` / `judge_decision` | 486 次 | 每次回合都启动 sidecar + judge 判定 |
| **`memory_injected`** | **仅 14 次** | 真正注入回话的极少 |
| 提取完成 `extraction_complete` | 仅 15 次 | 提取也极低 |

**结论**：管线每回合都全量运行（嵌入+过滤+sidecar+judge），但产出（注入/提取）不足 3%。即"高投入、低产出"，召回管线大部分计算白费。

### 1.2 80% 的判定在无 LLM 下降级运行
- `judge_decision` 486 次中 **389 次（80%）`no_llm=true`**
- `cadence_carry`（131 次）几乎全部伴随 `no_llm: true`，即因为重排所需 LLM 不可用而直接沿用上轮判定或跳过
- 17 次 `error` 事件全部为 **"Sidecar completion via active provider failed"**

**结论**：sidecar 重排（cross-encoder rerank）依赖 LLM，但多数回合该 provider 不可用，导致重排功能名存实亡，召回质量得不到保障。

### 1.3 记忆置信度单向衰减
| 指标 | 数值 |
| --- | --- |
| 累计 `boosted` | 74 |
| 累计 `decayed` | **2931** |
| 有 boost 的记录数 | 仅 27/486 |

**结论**：候选记忆几乎总是被衰减、极少被增强。这会导致记忆置信度持续走低，最终被 GC 清理，即使它们对当前任务相关。衰减逻辑缺少"正向相关增强"触发条件，是策略缺陷。

### 1.4 嵌入延迟高
- `embedding_complete` 延迟: min=158ms max=**8736ms** avg=704ms
- **>500ms 占 59%，>1000ms 占 18%**

**结论**：本地嵌入（minilm-l6-v2）在 7.5GB 内存 / UHD620 集显的机器上每次召回平均 0.7s，高时 8.7s，会拖慢每回合响应，进一步降低用户使用记忆的意愿。

### 1.5 候选命中饱和
- embedding hits 分布: 10 命中出现 90 次，2 命中 20 次，avg=8.5
- 候选过滤去重效率：940 → 705，仅去掉 25% 重复

**结论**：候选池总是取满 10 个，说明记忆库内高相似度条目多（重叠），但多数是冗余/噪声条目。

---

## 2. 结合存储数据的问题定位

### 2.1 噪声条目污染召回
176 条记忆中 **9 条碎片噪声**（`jq '.routes[]'`、`pipelines`、`服务名`、`85fa7777a`、`CONTENT`、`Commit` 等 <15 字符），来自 LLM 增量提取，无语义价值却参与召回打分，稀释真正相关记忆。

### 2.2 关系网络缺失，图遍历召回失效
- 10 张有数据图中，**5 张边=0**（jcode、v2-test2、xxl-job、AndClaw早期等全为孤立节点）
- 仅 schedule、kaneo、AndClaw 三图有 `derived_from`/`relates_to` 关系
- `has_tag` 边覆盖极不均衡（部分记忆打 `assistant`/`tool` 标签，多数无标签）

**结论**：当前召回基本只靠向量相似度，图结构（BFS 遍历、标签关联）未发挥效用。

### 2.3 向量分块与 FTS 检索未启用
- 所有 `_chunks` 表为空 → 未用向量分块
- 仅 2 张图（global、9a47e276）有 FTS5 全文索引

**结论**：召回退化为纯向量相似度，无法做关键词精确匹配或分块级检索。

### 2.4 记忆分散 + 空图残留
- 22 张空图（纯表结构），历史项目残留
- 项目哈希图多而散，跨项目记忆（如 AndClaw 有两个图）无法聚合

---

## 3. 优化方向建议

### 方向 A：降低召回管线开销（高优先级）
**问题**：每回合全量跑嵌入+sidecar+judge，产出<3%。
**建议**：
1. **按需触发召回**：引入"记忆相关度预筛"，只有当前回合与已存记忆存在语义重叠（如嵌入粗筛得分>阈值）才启动完整 judge/sidecar，而非每回合全跑。
2. **缓存嵌入结果**：相同/相似 turn 复用上次嵌入向量，避免重复计算。
3. **降级策略重构**：80% 无 LLM 时，直接采用向量相似度 + 图遍历评分作为 fallback，而非 `cadence_carry` 沿用旧判定或跳过，保证有可用的召回结果。

### 方向 B：修复置信度单向衰减（高优先级）
**问题**：2931 次衰减 vs 仅 74 次增强，记忆置信度被系统性压低。
**建议**：
1. 引入**正向增强触发**：当候选记忆与当前回合内容匹配度高（向量相似度>阈值）或用户后续行为验证了该记忆（如引用/执行），应 `boosted` 而非一律 `decayed`。
2. **修正衰减率**：区分"未被命中"（可衰减）与"被命中但未选"（应保持或微增），避免高相关记忆被误衰减后 GC 清理。
3. GC 前复核：`GcArchived` 清理前对候选做一次相关性复核，避免误删仍在用的记忆。

### 方向 C：清理噪声与冗余（高优先级，见效最快）
**问题**：9 条碎片噪声 + 高相似度重叠污染召回。
**建议**：
1. **提取侧过滤**：`llm_extraction` 后增加内容质量门控（content 长度、是否含可辨识语义、是否非纯标识符），碎片条目不写入。
2. **入库去重**：强化 `dedup_reinforce` 规则，相似度>阈值（如 0.92）时合并而非新增，减少候选池重叠（当前去重仅 25%）。
3. **噪声清理脚本**：批量 `forget` 掉 content<15 字符且无标签、访问次数为 0 的条目。

### 方向 D：补齐图关系与检索能力（中优先级）
**问题**：5 张图 0 边，向量分块/FTS 未启用。
**建议**：
1. **运行聚类活动**：`Activity::InCluster` 用 embedding 自动聚类并建 `in_cluster`/`relates_to` 边，填充 jcode、v2-test2 等 0 边图。
2. **启用 FTS5**：为活跃项目图建全文索引，支持关键词精确匹配与向量+BM25 混合召回。
3. **聚合分散记忆**：统一 AndClaw 等跨目录项目的 store key，或提供跨图检索合并机制。

### 方向 E：处理性能瓶颈（中优先级）
**问题**：嵌入平均 704ms、峰值 8.7s，拖慢响应。
**建议**：
1. **异步预取**：在用户输入时后台预计算嵌入，回合开始时已就绪。
2. **模型/量化优化**：评估更小的嵌入模型或 int8 量化，或改用 GPU（若可用）加速。
3. **结果缓存**：高频查询向量化缓存，避免重复嵌入相同上下文。

---

## 4. 优先级排序与预期收益

| 优先级 | 方向 | 预期收益 |
| --- | --- | --- |
| P0 | B（修复单向衰减）+ C（清理噪声） | 立即提升召回精度、减少 GC 误杀，见效最快 |
| P0 | A（降管线开销） | 减少 80% 无意义计算，加速响应 |
| P1 | D（补齐图关系+检索） | 激活图遍历召回，长期提升记忆效用 |
| P1 | E（性能优化） | 改善体验，降低 0.7s 平均延迟 |
| P2 | 空图/分散清理 | 减少冗余，提升库整洁度 |

---

## 5. 根因结论（已从 `jcode-2026-08-28.log` 验证）

### 5.1 `no_llm` 降级的根因：MiniMax 直连端点 + 限流，非 sidecar 未启动
- `memory_sidecar_enabled=true`，sidecar **确实随 daemon 运行**（无独立 sidecar 进程，是内存内逻辑）。`llm_backend_available()` 走 `active_provider_fork()` 分支返回 `true`，因为确实有活跃 provider（当前会话用 OpenRouter/MiniMax/deepseek）。
- **模型命名空间不兼容（84 次 fallback warn）**：active provider 是 OpenRouter 槽但实际指向**直连 MiniMax 端点**（`https://api.minimaxi.com/v1`，`OPENAI_COMPAT_API_KEY`）。`safe_model_for_provider()` 检测到 provider 携带 `anthropic/claude-sonnet-4`（OpenRouter 命名空间）与直连 MiniMax 不兼容，改写为默认 `MiniMax-M3` 以防 HTTP 400 "Model Not Exist"。
- **MiniMax 429 限流（9 次，全部命中 memory judge 时段）**：`session_bear/peacock/poodle` 主会话本身就是 MiniMax 模型（`mod:MiniMax`），并发跑满配额。memory sidecar 复用同一 `active_provider_fork()` 的 MiniMax provider，judge 调用被打回 429 → 今日 3 次 `all_judges_failed`（11:04、11:17、12:34）全部对应限流时段。

**结论**：80% `no_llm` 拆解（486 次 judge_decision）：
| 类别 | 次数 | 占比 | 说明 |
| --- | --- | --- | --- |
| `cadence_carry` | 284 | 58% | **正常**，重排有节流节奏，非降级 |
| `all_judges_failed` | 105 | 22% | **真降级**，根因 = MiniMax 直连 + 429 限流 |
| `judge_ran` | 97 | 20% | 正常判定 |

所以「80% 降级」的表述**过高**：实际真降级约 22%，且集中在 MiniMax 主模型会话上。历史 105 次 `all_judges_failed` 多为同根因。

### 5.2 记忆注入仅 14 次的根因
- 主因是 judge 频繁降级/失败：`all_judges_failed` 时 `"surfacing nothing (caller carries verified set)"`，不产生新注入。
- 次因是 `cadence_carry` 频繁沿用 `re-surfacing 0 consensus-verified memories`（今天 25 次全为 0），即重排因节流/降级返回空集，注入源头枯竭。
- 真正成功的 `Final extraction` 很少：今天仅 2 次（bear 存 40 条、dragon 失败），因 MiniMax 429 导致 18:17 `Final extraction ... failed: Sidecar completion via active provider failed`。

### 5.3 根因修复建议（新增）
1. **sidecar 与主会话解耦**：memory judge/rerank 不应复用主会话的 MiniMax provider（会被主会话的流量挤爆 429）。应单独配置一个低并发、稳定的 memory 专用模型（如 deepseek 直连或独立 API key），见 `agents.memory_model` 配置项。
2. **对 429/瞬时错误重试**：`complete_via_provider` 对限流增加退避重试（当前直接失败进入 `all_judges_failed`）。
3. **降级 fallback 而非跳过**：`all_judges_failed` 时应回退到纯向量相似度排序，而不是 `surfacing nothing`，保证召回不断流。
4. **修正节流统计口径**：`cadence_carry` 不应计为 `no_llm` 降级，当前 58% 被误报为降级，掩盖了真实 22% 的限流问题。
