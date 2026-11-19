# 工程本体基础层（Engineering Ontology）

本文档描述 `jcode-memory-types::engineering_ontology` 的落地范围、分层设计和接入方式。

## 1. 定位

这不是 BFO、ISO/IEC 24744、OMG Essence 或 SWEBOK 的全量导入，而是一个**可编译、可校验、可扩展的工程本体 profile**：

- **BFO-lite**：每个类带一个上层范畴，区分独立连续体、一般依赖连续体、过程、质量等。
- **ISO/IEC 24744 形状**：Session、Turn、Task、WorkUnit、Action、Observation、Artifact 组成 endeavour / work unit / work product 骨架。
- **OMG Essence Alpha**：Requirements、Software System、Work、Team、Way of Working 等 alpha 及其状态序列。
- **SWEBOK/SKOS 关注点**：requirements、design、construction、testing、maintenance、configuration_management、engineering_management、software_quality、engineering_economics。
- **PROV-O 形状溯源**：每个实体和关系都带 source、locator、activity、derived_from、confidence、recorded_at。
- **SHACL 形状校验**：类、属性、必填、值类型、domain/range、时间窗、基数。

运行时表示是属性图 `EngineeringGraph`。OWL DL 推理不在运行时使用。

## 2. 模块 API

```rust
use jcode_memory_types::engineering_ontology::{
    default_engineering_ontology, EngineeringEntity, EngineeringGraph, EngineeringRelation,
    CLASS_ERROR, CLASS_FILE, CLASS_TASK, REL_BLOCKED_BY, REL_DEPENDS_ON,
};
```

核心类型：

- `EngineeringOntology`：类、关系签名、关注点、alpha 状态。
- `EngineeringGraph`：实体和关系实例，带 `validate` 与 `shortest_path`。
- `EngineeringEntity`：typed entity。
- `EngineeringRelation`：typed relation。
- `EngineeringProvenance`：PROV-O 形状溯源。
- `EngineeringValidationReport`：错误和警告。

## 3. 示例

```rust
use jcode_memory_types::engineering_ontology::*;
use serde_json::json;

let ontology = default_engineering_ontology();

let mut graph = EngineeringGraph::new();
graph.insert_entity(
    EngineeringEntity::new(CLASS_TASK, "task:#157", "修复 memory 校验")
        .with_property("title", json!("修复 memory 校验"))
        .with_property("status", json!("in_progress")),
);
graph.insert_entity(
    EngineeringEntity::new(CLASS_FILE, "file:crates/jcode-base/src/memory.rs", "memory.rs")
        .with_property("path", json!("crates/jcode-base/src/memory.rs")),
);
graph.insert_entity(
    EngineeringEntity::new(CLASS_ERROR, "error:EO11", "EO11")
        .with_property("signature", json!("EO11")),
);

graph.insert_relation(EngineeringRelation::new(
    "task:#157",
    REL_BLOCKED_BY,
    "error:EO11",
));
```

校验：

```rust
let report = graph.validate(&ontology);
assert!(report.is_valid(), "{:?}", report.errors());
```

路径查询：

```rust
let path = graph.shortest_path("task:#157", "error:EO11", 3, &[REL_BLOCKED_BY]);
```

## 4. 校验码

| 代码 | 含义 |
|---|---|
| EO01 | ontology id 为空 |
| EO02 | ontology version 非正数 |
| EO10 | 类父类不存在 |
| EO11 | 类属性重复声明 |
| EO12 | 属性引用未知类 |
| EO13 | 类使用未知关注点 |
| EO14 | 类继承环 |
| EO20 | 关系引用未知类 |
| EO21 | 关系使用未知关注点 |
| EO22 | 关系 domain/range 为空 |
| EO30 | alpha 没有状态 |
| EO31 | alpha 状态重复 |
| EG01 | 实体 id 为空 |
| EG02 | 实体名称为空 |
| EG03 | 实体 confidence 越界 |
| EG04 | 实体 provenance confidence 越界 |
| EG05 | 实体 valid_to 早于 valid_from |
| EG10 | 实体使用未知类 |
| EG11 | 缺少必填属性 |
| EG12 | 属性值类型不匹配 |
| EG13 | 使用未声明属性（警告） |
| EG14 | active 实体超出有效时间窗（警告） |
| EG20 | 关系 id 为空 |
| EG21 | 关系 confidence 越界 |
| EG22 | 关系 provenance confidence 越界 |
| EG23 | 关系 valid_to 早于 valid_from |
| EG24 | 关系源实体不存在 |
| EG25 | 关系目标实体不存在 |
| EG26 | 关系类型不存在 |
| EG27 | 关系 domain 不接受源类 |
| EG28 | 关系 range 不接受目标类 |
| EG30 | 关系从同一源超过最大基数 |

## 5. 与后续能力的接口

- **Episode 抽取**：`bash`、`edit`、`apply_patch`、`todo`、`selfdev` 的轨迹映射到 `Action`、`Command`、`File`、`Error`、`Change`、`Test`。
- **压缩锚点**：`Graph Memory State` 使用 `task:#`、`file:`、`error:`、`command:` 等稳定 id。
- **路径查询**：`shortest_path` 是确定性原语，后续可加 Yen k-best、边权和 Hub 惩罚。
- **计划生成**：todo DAG 映射到 `Task` / `WorkUnit`，关系映射到 `depends_on`、`blocked_by`、`implements`、`verifies`。
- **记忆层**：现有 `MemoryGraph` 保留事实、偏好、约定，不强行改造成领域实例图。

### 5.1 Episode 抽取器

`jcode-base::engineering_episode::extract_engineering_episode` 是第一个生产方：

```rust
use jcode_base::engineering_episode::extract_engineering_episode;

let extraction = extract_engineering_episode(session_id, &messages, Some(working_dir));
let graph = extraction.graph;
let warnings = extraction.warnings;
```

当前确定性映射：

| 工具 | 产物 |
|---|---|
| `bash` | `Action -> calls -> Command`，错误结果生成 `Error` |
| `edit` / `replace` / `write` / `apply_patch` | `File`、`Change`，`Action -> modifies -> File` |
| `read` | `Action -> uses -> File` |
| `todo` | `Task`，`Action -> implements -> Task` |
| 任意工具结果 | `Observation`，`Action -> produces -> Observation` |
| 错误结果 | `Action -> causes -> Error`，错误签名做稳定 id |

该抽取器不使用 LLM，未知工具只生成 `Action`，不生成猜测性的文件或命令边。

### 5.2 压缩摘要锚点

`render_graph_memory_state(messages, max_anchors)` 会从同一批消息生成确定性锚点，并在文本摘要末尾追加：

```text
## Graph Memory State

- `file:src/lib.rs` | file | src/lib.rs
- `command:...` | command | cargo test -p jcode
- `error:...` | error | error[E0425]: cannot find function
```

该块只包含稳定 id，后续 `shortest_path` 或 `memory recall` 可以直接解析。Session、Turn、Observation 不进入锚点段，因为它们已由摘要正文表达。

追加前会做 shrink proof：如果模型返回的摘要本身已经不小于被压缩消息的总字符数，或者加上锚点后会不再变小，就跳过锚点段并记录 warning。锚点段自身上限为 4096 字符，超出时按整行截断。这样锚点不会让一次本来失败的压缩变得更差。

### 5.3 持久化与锚点回读

`jcode-base::engineering_store::EngineeringStore` 把每个 session 的属性图作为一个 JSON 图保存到：

```text
<JCODE_HOME>/engineering/graphs.sqlite
```

默认路径是 `~/.jcode/engineering/graphs.sqlite`。表 `engineering_graphs` 以 `session_id` 为主键，同时保存 ontology id/hash、实体数、关系数、`created_at` 与 `updated_at`。当前版本使用 JSON 图而不是规范化三元组表，原因是 Episode 图规模小，路径查询在内存中完成；后续需要跨 session 图查询时再拆表。

压缩完成、摘要正式生效时，以及会话正常关闭或崩溃落盘时，Agent 会重新抽取该 session 的图并 upsert：

```rust
let extraction = extract_engineering_episode(session_id, &messages, working_dir);
store.save_graph(session_id, working_dir, &extraction.graph)?;
```

回读通过 `memory` 工具：

```json
{"action":"recall","anchor":"file:src/lib.rs","depth":2,"limit":10}
```

返回所有包含该锚点的 session 子图，包含 `depth` 跳内的实体和两端都在子图内的关系。`limit` 是扫描的 session 数上限，`depth` 默认 2。`file:`、`error:`、`command:`、`task:` 都使用同一套稳定 id。

```rust
let recalls = store.recall_anchor("error:0123abcd", 2, 10)?;
```

锚点查询先走 `engineering_entity_index(session_id, entity_id)` 定位候选 session，再在内存中精确过滤并做 BFS。索引在每次保存时随图同步写入，旧库打开时会自动回填。关系仍只在内存中遍历；如果需要跨 session 的关系级检索，再增加 `engineering_relations(session_id, from_id, to_id)` 索引表。

保留策略默认保留最新两个非空 session、任何 30 天内的 session，并始终把只有 Session 节点的空图标为候选：

```bash
jcode memory storage engineering-audit
jcode memory storage engineering-prune --apply
```

`engineering-audit` 只读；`engineering-prune` 默认 dry-run，只有 `--apply` 才删除候选行。当前策略不会裁剪大型活跃 session，只清理空图和超过 30 天且不在最新两个之内的历史图。

工程图默认以 gzip 压缩 BLOB 保存，并维护 `engineering_entity_index` 供锚点回读。存量 JSON 行可用：

```bash
jcode memory storage engineering-compact
jcode memory storage engineering-compact --apply
```

压缩只替换更大的旧 JSON，不会改变实体、关系或锚点语义。

### 5.4 带权路径与 k-best

`shortest_path` 保留原来的无权 BFS 语义，用于只需要连通性的调用。需要排序或解释时使用：

```rust
let path = graph.weighted_shortest_path("file:src/lib.rs", "error:0123", 4, &[]);
let alternatives = graph.k_shortest_paths("file:src/lib.rs", "error:0123", 4, &[], 3);
```

边权为 `(1 + 2 * (1 - confidence)) * property["weight"]`，`weight` 缺省 1.0。`paths[i].cost` 是边权和，`confidence` 是路径上最弱边的置信度，`provenance` 按边顺序返回 PROV-O 形状记录。`k_shortest_paths` 使用 Yen 算法，结果按 cost、节点序列稳定排序。

## 6. 非目标

- 不实现 OWL DL 推理。
- 不导入完整 BFO、ISO/IEC 24744、Essence 或 SWEBOK 文本。
- 不自动把 LLM 提议的实体或关系晋升为事实。
- 不替代现有 `MemoryGraph`。

## 7. 测试

```bash
scripts/dev_cargo.sh test -p jcode-memory-types engineering_ontology --lib
```
