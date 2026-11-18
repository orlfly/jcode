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

## 6. 非目标

- 不实现 OWL DL 推理。
- 不导入完整 BFO、ISO/IEC 24744、Essence 或 SWEBOK 文本。
- 不自动把 LLM 提议的实体或关系晋升为事实。
- 不替代现有 `MemoryGraph`。

## 7. 测试

```bash
scripts/dev_cargo.sh test -p jcode-memory-types engineering_ontology --lib
```
