# Agent Benchmark 约定

状态：📋 设计中。本文档先于 harness 定稿，harness 见 `scripts/benchmark.py`，
结论落在 `docs/evidence.md`。

相关：`scripts/constraint_experiment.py`（约束保留对照实验）、`docs/evidence.md`（结论落点）。

## 1. 定位

| 项 | 约定 |
| --- | --- |
| 评什么 | Agent 的**行为质量**：工具选择、参数准确、步数、错误恢复、约束遵守、长上下文召回 |
| 不评什么 | 答案质量（引用可溯源、SQL 可编译这类由智能问数的五层评测负责）与模型本身的编码水平 |
| 形态 | 独立脚本 `scripts/benchmark.py`，与 `constraint_experiment.py` 平行：互不 import、各自能单独跑 |
| 复用 | 数据模型：record 字段、JSONL 落盘、`--resume` 跳过已跑、`--self-test` 离线自检、脚本化判定 + 聚合表 |
| 不新建 | 不引入新依赖、不加联网 CI job（CI 只跑 `--self-test`） |
| 联网 | 真跑需要 `LLM_API_KEY`；判定逻辑全部离线可复算 |
| 轮数上限 | `--max-turns` 不传时按类别默认（A/B 类 8、C 类 6），是安全网而不是评测项（效率由 `steps` + `within_steps` 度量）；每次 run 记下实际用的上限（`max_turns`），被截断的 run 另标 `observations.hit_max_turns` 与 `stopped_reasons`，不静默记成任务失败 |

独立成脚本而不是并进实验脚本的原因：实验脚本的会话结构是「声明约束 → 填充 → 探针」一种形态，
benchmark 有 12 种任务形态，混在一起会让两边的 profile 语义、续跑键、汇总表全都互相污染。

## 2. 分类：category 与 capability 解耦

两个**正交**维度，各自打标：

- `category`（标量，任务形态与夹具形状）：`retrieval` / `edit_exec` / `long_context`
- `capabilities`（数组，被考察的行为能力）：`tool_selection` / `param_accuracy` / `efficiency` /
  `error_recovery` / `constraint_retention` / `long_context_recall`

**不用 `Step.kind` 当分类轴**，三条理由：

1. `kind` 描述的是「这一步在会话里扮演什么角色」（declare / fill / probe），属于构造细节；
   A/B 两类任务根本没有 declare 与 probe 步骤，套上去只会得到空类别。
2. 一个任务只属于一个 `category`，但可以同时评多个 `capability`——B3 既评 `error_recovery`
   也评 `param_accuracy`，B1/B2 都评 `param_accuracy` 与 `efficiency`。用单一枚举表达不了这种交叉。
3. 聚合要能按任一维度切：`by_category` 回答「哪种形态的任务难」，`by_capability` 回答「哪个能力弱」。
   揉成一个枚举后只能得到一张语义混合的表。

harness 内部仍有 `Step(label, text)`，`label ∈ setup/declare/fill/probe`，**只用来跑完后定位该读哪条回复**，
不参与分类、不写进 record。

## 3. 证据位置

| 内容 | 位置 | 说明 |
| --- | --- | --- |
| 逐次记录 | `docs/evidence/benchmark_runs.jsonl` | 一行一次；`(task_id, group, run_id, profile)` 是续跑去重键 |
| 汇总结论 | `docs/evidence.md`「用例六：Agent 行为 benchmark」 | harness 只打印 Markdown 表；落库由人确认后粘贴 |
| 复现命令 | `docs/evidence.md` 与 `README.md` 的评测段 | `--self-test` 与真跑两条命令都要写进去 |
| 不入库 | tempfile 工作区、模型回复全文 | record 只留判定结果与必要的短摘录 |

## 4. 与 constraint_experiment 的字段边界

**共用字段**（同名必须同义，改一边要改两边）：

| 字段 | 含义 |
| --- | --- |
| `run_id` / `group` / `profile` / `seed` | 第几次 / on·off / standard·stress / 复现种子 |
| `task_success` / `checks` / `violated` | 全部检查是否通过 / 检查明细 / 违反条数 |
| `constraints_total` / `preserved` / `preserved_verbatim` / `replenished` | 声明条数 / 代号仍在上下文 / 代号+原文都在 / 自愈补录数 |
| `codes_recalled` | 探针能答出的代号数 |
| `tiers_triggered` / `summary_calls` / `summary_messages` | 四层触发次数 / Tier 4 次数 / 最终摘要消息数 |
| `llm_calls` / `duration_s` | LLM 轮数 / 耗时 |

**benchmark 专有**：`task_id`、`category`、`capabilities`、`allowed_tools`、`required_tools`、
`tools_used`、`tool_selection_ok`、`params_ok`、`steps`、`escalations`、`aborted`、`max_turns`、`observations`
（`observations` 放软指标，例如「是否走到过 write_file 被拒的路径」，不参与 `task_success`）。

**experiment 专有**：无。它的 `profile` 调的是约束条数（15 / 40），benchmark 的 `profile` 调填充轮数；
两档名字一致、调校对象不同，属于同名但不同对象的例外，已在上一节与本节各注明一次。

三条硬规则：

1. benchmark 不新增与共用字段语义重叠的新名字（不另立 `kept_codes`，直接用 `preserved`）。
2. experiment 不引入 `task_id` / `category`；benchmark 不引入约束专用的运行期开关字段。
3. 两边各自读自己的 JSONL，缺字段按默认值处理，互不要求对方升级。

## 5. 待确认项（4）

| 编号 | 问题 | 现状 / 证据 | 建议默认值 | 影响面 |
| --- | --- | --- | --- | --- |
| Q1 | A4 的 `steps` 上限 | A4 是只读任务，正常路径 1 次 `read_file`；文件较长时模型可能重读 | `steps ≤ 2`（1 次读 + 允许 1 次重读），另给 `--strict-steps` 收紧到 1 | A4 的 `efficiency` 判定松紧 |
| Q2 | `summary_messages` 是否已存在 | **已存在**：`scripts/constraint_experiment.py:339` 写入、aggregate 取 max；待确认的是它是否作为 C 类硬判据 | 沿用同名同义，并把 `summary_messages == 1` 作为 C 类硬判据 | C 类 `task_success` 是否被摘要叠加问题直接判负 |
| Q3 | A/B 类是否只跑 `group=on` | A/B 任务不声明约束，`off` 组不产生新信息，纯翻倍成本 | A/B 固定 `group=on` 且不建 `ConstraintStore`；只有 C 类跑 on/off 对照 | 真跑成本；以及「约束机制是否影响正常路径」这条信度检查要不要做 |
| Q4 | B3 的 `escalations` 如何观测 | L3 的钩子是 `AgentLoop(on_tool_failure=...)`，调用次数即升级次数；`loop._escalated` 是私有状态 | 用钩子计数（固定返回 True 只计不拦），`aborted = stopped_reason == "user_aborted"`；不读私有字段 | B3 的 `error_recovery` 判定方式 |

Q4 附带说明：钩子返回 False 的中止路径不单列用例，由 `tests/test_failure_escalation.py` 覆盖，
benchmark 只测「能恢复」这一侧。

### 5.1 已拍板（首次全量 `on` 档跑完后修订）

1. **A/B 类的 JSON 解析放宽**：允许整段被一层 ``` 围栏包住（`parse_json_lenient`）。
   C 类保持严格——C1 的「必须是合法 JSON」、C2 的「不得出现代码围栏」本身就是被考察的约束，
   放宽会把该测出来的违规洗掉。影响：A2 由「格式违规」改为按内容判。
2. **B4 的步数上限 4 → 6**，与 B1–B3 对齐。效率已经由 `steps` + `within_steps` 度量，
   不该因为一条任务要试探 shell 就比同类更苛刻。
3. **「按 group」汇总只统计 `constraints_total > 0` 的行**（即 C 类）。A/B 不声明约束，
   混进去会把保留率的分母与语义一起搅乱。
4. **`observations` 增加 `escalated_tools`**：L3 钩子只收到一段文本，工具名从文本里解，
   解不出记 `unknown`。这是补全 Q4 的可观测性。
5. **轮数上限按类别给**：A/B 类 6 → 8，C 类保持 6。A/B 是「一件事做完就收尾」，需要留出收尾轮；
   C 类每步只读一个文件，6 轮足够。上限不替代效率度量（`steps ≤ steps_limit` 另有其表），
   每次 run 把实际上限写进 `max_turns`，跨配置比较时能看出差异。

## 6. 任务清单

12 个任务的**唯一事实来源是代码**（`scripts/benchmark.py` 的 `build_tasks()`）。文档不复制一份，
避免两边漂移。要看清单就现打：

```bash
.venv\Scripts\python scripts/benchmark.py --list-tasks
```

输出两段：总览表（id / category / capabilities / 必需与允许工具 / 步数上限 / 会话步数 / 跑哪些 group），
以及每个任务的会话脚本（`declare` / `fill` / `probe` 逐条列出）。命令离线可跑，不需要 `LLM_API_KEY`。

提示词里的随机串（例如 `notes/part03.txt` 里待替换的记号）来自当次夹具，每次运行都不同——
这是设计的一部分：代号与内容无关，模型推不出来，只能靠真的读到（见 §2）。

分布：retrieval 4 个（A1–A4）、edit_exec 4 个（B1–B4）、long_context 4 个（C1–C4）。

## 7. 实施顺序

1. 本文档定稿（含 4 个待确认项拍板）
2. 按本文档重出 12 个任务的清单
3. 写 `scripts/benchmark.py`（`--self-test` 必须离线通过）
4. 真跑：`--group both --runs N`
5. 结论进 `docs/evidence.md` 用例六，复现命令同步进 README