# Jcode Memory 模块存储数据分析报告

采集时间: 2026-08-28 17:42 CST
采集对象: `/home/cc/.jcode/memory/backend-sqlite/gvec.sqlite`
运行版本: jcode v0.81.63-dev (6d3821d68)

---

## 1. 存储架构总览

记忆系统采用 **SQLite + gvec 图数据库扩展** 存储，单文件 `gvec.sqlite`（主库 2.39MB + WAL 4.17MB）。每个逻辑存储域（全局或某个项目）映射为库内一个**命名图**，图名格式 `g_<sanitized_store_key>`。

### 1.1 图命名规则（源码 `sqlite_gvec.rs:134-152`）
- 全局域: `g_global`
- 项目域: `g_project_<16位十六进制哈希>`，哈希来自 `project_store_key()`（`memory.rs:2702`），即对项目目录路径做 Rust `DefaultHasher` 后取 `project:{:016x}`，再 sanitize 成 `project_<hash>`。

### 1.2 每张图的数据表
| 表 | 作用 |
| --- | --- |
| `<graph>_nodes` | 节点（记忆条目 / 标签 / 实体），`labels` + `properties`(JSON) |
| `<graph>_edges` | 有向边，`source`/`target`/`edge_type`/`properties` |
| `<graph>_chunks` / `_rowids` | 向量分块索引（当前全部为空，未启用向量分块） |
| `jcode_fts_<graph>` | FTS5 全文索引（仅 global 和 9a47e276 两张图有） |

### 1.3 节点标识
- 记忆条目: `mem_<毫秒时间戳>_<随机数>`（如 `mem_1787891658619_5512811802090022058`）
- 标签: `tag:<name>`
- 字符串 id 直接存在 `properties["__jcode_id"]`，整数 rowid 作为内部句柄。

---

## 2. 数据规模统计

### 2.1 总体
| 指标 | 值 |
| --- | --- |
| 图数量 | 32（其中 10 张有数据，22 张空） |
| 总节点 | 192 |
| 总边 | 224 |
| 活跃节点 | 176（91.7%） |
| 非活跃节点 | 16（8.3%） |
| 有访问记录节点 | 156 |
| 总访问次数 | 231 |
| 强化(reinforcement)总数 | 143 |
| 空 content 节点 | 16（全部为 Tag 节点，正常） |

### 2.2 按类别 category
| 类别 | 数量 | 占比 |
| --- | --- | --- |
| fact（事实） | 138 | 71.9% |
| preference（偏好） | 22 | 11.5% |
| correction（修正） | 14 | 7.3% |
| entity（实体） | 2 | 1.0% |
| tag/other（标签等） | 16 | 8.3% |

### 2.3 按信任度 trust
| 信任度 | 数量 |
| --- | --- |
| high | 96 |
| medium | 77 |
| low | 3 |
| (无) | 16 |

### 2.4 按提取方法 method
| 方法 | 数量 |
| --- | --- |
| llm_extraction（LLM 自动提取） | 173 |
| user_stated（用户明示） | 3 |
| (无) | 16 |

### 2.5 按来源 source
| 来源 | 数量 |
| --- | --- |
| incremental（增量提取） | 58 |
| session_bear（kaneo 项目） | 31 |
| session_mizaru（jcode 项目） | 22 |
| session_rat（AndClaw） | 9 |
| session_bird/chick/chicken/cat（v2-test2） | 18 |
| 其余 session | 38 |

---

## 3. 各图（项目）数据分布

| 图 | 节点 | 边 | 对应项目 | 主要来源 |
| --- | --- | --- | --- | --- |
| `g_global` | 3 | 1 | 全局 | peacock(schedule) |
| `g_project_8add22fa186e85db` | 56 | 77 | kaneo | bear + incremental |
| `g_project_14382533db35c776` | 36 | 75 | schedule | incremental + wyvern |
| `g_project_3b3ebe113061c41c` | 24 | 0 | jcode | mizaru |
| `g_project_c609e5addc13ce46` | 24 | 71 | AndClaw | incremental + rat |
| `g_project_bf2abad6197c5c19` | 18 | 0 | v2-test2 | bird/chick/chicken/cat |
| `g_project_5c542362a15e3953` | 13 | 0 | AndClaw(早期) | dragon/hatchling/unicorn |
| `g_project_9a47e276bfce8ae0` | 11 | 0 | ~/.jcode 自身 | humpback |
| `g_project_5df4e7b85c6e82c0` | 5 | 0 | ~/home | duckling |
| `g_project_4b5dfef0867a870b` | 2 | 0 | xxl-job | tulip |

> 注：`g_project_5c542362a15e3953` 与 `g_project_c609e5addc13ce46` 均含 AndClaw 相关记忆，可能是同一项目不同工作目录哈希（如 `/opt/workspace/AndClaw` 与 `/data/...` 或不同路径）产生的两个图。

---

## 4. 全局记忆（g_global）详细内容

全局域仅 3 个节点，均为**策略类偏好**，来源是 schedule 项目的 peacock session：

| id | 类别 | 内容摘要 |
| --- | --- | --- |
| `spring-mvc-shadow-controller-pattern` | preference | Spring MVC shadow 控制器会完全顶替原版全部 @RequestMapping，必须完整复制原版 endpoint，不能"仅新增端点"。修复 #48330 时踩坑（pageList/loadById/delete 全 404） |
| `memory-hygiene-verify-before-remember` | preference | 更新全局 memory 前必须 verify 事实准确性，矛盾时应 forget 旧条目再 remember 修正版。访问 3 次，带 `assistant` 标签 |
| `tag:assistant` | Tag | 标签节点，count=1 |

全局记忆边：`memory-hygiene-verify-before-remember -[has_tag]-> tag:assistant`

---

## 5. 图结构分析

### 5.1 边类型分布
| 边类型 | 数量 | 语义 |
| --- | --- | --- |
| `derived_from` | 大量 | 程序性知识由事实派生（如 shadow 修复经验由多个 fact 派生） |
| `relates_to` | 大量 | 语义关联（带 weight） |
| `has_tag` | 若干 | 记忆挂标签 |

### 5.2 标签体系
| 标签 | 出现图 | count |
| --- | --- | --- |
| `assistant` | global, 8add22fa, 14382533, c609e5ad | 25 |
| `1fce7c214` | c609e5ad | 7 |
| `think` | c609e5ad | 5 |
| `tool` | 14382533 | 3 |
| `server` | c609e5ad | 3 |
| `chrome` | 8add22fa | 2 |

### 5.3 图密度
- `g_project_8add22fa186e85db`（kaneo）: 56 节点 / 77 边，密度最高，关系网络最丰富
- `g_project_14382533db35c776`（schedule）: 36 节点 / 75 边，同样高密度
- `g_project_c609e5addc13ce46`（AndClaw）: 24 节点 / 71 边
- 其余图（jcode、v2-test2、xxl-job 等）**边为 0**，只有孤立节点，未建立关系网络

---

## 6. 记忆内容主题分析

### 6.1 全局/跨项目主题
1. **Spring MVC shadow 控制器模式**（#48330 修复经验）— 全局记忆，跨项目复用
2. **记忆卫生**（verify-before-remember）— 全局记忆，元认知策略

### 6.2 schedule 项目（14382533）
- xxl-job-admin 3.5.0 shadow 类路径（XssUtil/StringTool/Consts/XxlJobRegistryMapper）
- Higress Console API CRUD routes（/v1/routes，basic auth admin:Kingsware@123）
- HIGRESS_EXTRA_ROUTES 非持久化问题
- Ki-Agents 前端经 Caddy 而非 Higress
- MUI TablePagination 中文国际化
- 端口/路由排查（8300/8301/28080/11002）

### 6.3 kaneo 项目（8add22fa）
- Kaneo 任务平台 API（POST/PUT /api/tasks，需 status 字段，Bearer Token）
- ki-agent-v2 微服务平台架构（OpenAPI 契约、MR 流程）
- AGENTS.md vs AGENT.md 加载规则
- 多 agent 协作测试项目 v2-test2 配置

### 6.4 jcode 项目（3b3ebe11）
- 紧急压缩（emergency compaction）触发阈值与修复
- Ollama 原生 /api/show capabilities 数组
- Chrome browser tool 22 个 schema action 实现
- selfdev 构建基础设施

### 6.5 AndClaw 项目（c609e5ad / 5c542362）
- adb shell 广播 TEST_ASR 注入假 ASR 事件
- JNI shim 17 个符号
- AndClaw MemoryGraph HTTP API（/api/memories，端口 5561）
- OpenSpec spec-driven 开发流程

---

## 7. 数据质量与潜在问题

### 7.1 值得注意的观察
1. **低质量/碎片化条目**：部分记忆 content 极短或无意义，如 `jq '.routes[]`、`pipelines`、`服务名`、`元数据 preserved.register.source`、`85fa7777a`、`CONTENT`、`描述当前屏幕 (screen_query_vision)`。这些是 LLM 增量提取的噪声，价值低。
2. **敏感信息存储**：`g_project_bf2abad6197c5c19` 中存有 Kaneo API key 明文（`FtTDqahhoRAFTKoUgpZbendjgHnZgLBksKPWGyLCMuiFPUNnqUzAdDZyAVMvGUiV`）和 GitLab 凭据说明，存在安全风险。
3. **空图冗余**：22 张图完全为空（0 节点 0 边），是历史项目残留，占用表结构但无数据。
4. **向量分块未启用**：所有 `_chunks` 表为空，说明当前未使用向量分块检索，仅依赖 FTS5 全文索引（且仅 2 张图有 FTS）。
5. **关系网络不均衡**：jcode、v2-test2、xxl-job 等项目的记忆全是孤立节点（0 边），未建立 derived_from/relates_to 关系，召回时无法利用图遍历。
6. **访问频率低**：多数节点 access_count=1，仅少数（如 Higress 相关）达到 5-6 次，说明记忆复用率不高。

### 7.2 强化（reinforcement）机制
- 143 次强化记录，集中在少数条目（如 `pipelines` 强化 5 次、`85fa7777a` 强化 8 次），但这些条目本身价值低，强化机制可能放大了噪声。

---

## 8. 架构设计要点（源码佐证）

### 8.1 三层设计（`ontology.rs`）
1. **Schema**：`Ontology`/`OntologyType`/`PropertyDef`/`RelationshipDef`/`LifecyclePolicy`
2. **Rules**：`Rule`/`Effect`/`Condition`，声明式规则在 remember/upsert/dedup/supersede/contradict/tag/link 事件上触发
3. **Activities**：`Activity`/`ActivityTrigger`/`ActivityStep`，定时或事件驱动（per_turn_relevance、periodic_extract、topic_change_extract、final_extract、gc_archived、summarize）

### 8.2 边类型（`graph.rs:92`）
`HasTag`(0.8)、`InCluster`(0.6)、`RelatesTo`(weight)、`Supersedes`(0.9)、`Contradicts`(0.3)、`DerivedFrom`(0.7)，各有 BFS 遍历权重。

### 8.3 存储后端（`sqlite_gvec.rs`）
- 单文件多图，WAL 模式支持跨进程并发读
- 进程内 `write_lock` Mutex 串行化写
- 字符串 id 存于 `properties["__jcode_id"]`

---

## 9. 建议

1. **清理噪声条目**：对 content 过短（<15 字符）或无实际语义的 fact 执行 `memory forget`，避免污染召回。
2. **敏感信息脱敏**：API key / 凭据不应明文入记忆库，建议改为引用环境变量或加密存储。
3. **清理空图**：22 张空图可安全清理（仅表结构，无数据）。
4. **补齐关系网络**：对 jcode、v2-test2 等 0 边项目，可运行聚类/关联活动建立 derived_from/relates_to 边，提升图遍历召回。
5. **评估向量分块**：当前未启用 chunks，若召回精度不足可考虑启用向量分块检索。
