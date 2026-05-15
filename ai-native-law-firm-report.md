---
title: "AI 原生律所赛道调研报告"
subtitle: "美国 YC、红杉系创业公司全景与商业模式分析"
date: "2026 年 5 月"
---

# AI 原生律所（AI-Native Law Firm）赛道调研报告

## 一、核心概念：跟传统 Legal Tech 的本质区别

过去 15 年的 Legal Tech 主流路径是**卖软件给律师**（SaaS 模式）—— Ironclad、Spellbook、Harvey、Casetext 都属于这一脉。

2026 年开始，硅谷出现了一个新物种：**AI-Native Law Firm**。它的核心定义是：

> **不是卖软件给律所，而是自己就是律所** —— 注册法律实体、买职业责任保险（malpractice insurance）、雇律师、对外交付法律服务（而非交付工具），AI 在内部作为生产力杠杆。

### 关键差异对比

| 维度       | 传统律所             | Legal Tech SaaS    | AI 原生律所            |
| ---------- | -------------------- | ------------------ | ---------------------- |
| 卖什么     | 法律服务（律师小时费）| 软件 license       | 法律服务（包干价/订阅）|
| 法律责任   | 律所承担             | 不承担（用户自负） | 律所承担               |
| 收入天花板 | 受律师人数限制       | 软件可无限扩展     | 软件+服务混合          |
| AI 角色    | 辅助工具             | 卖的就是 AI        | 内部生产引擎           |
| 单位经济   | 线性                 | 软件级             | 介于两者间，目标 40%+ 毛利 |

YC 在 2025 年的 Request for Startups 里直接喊话：**"start your own law firm, staff it with AI agents, and compete with existing law firms"** —— 这是这个赛道明确被 YC 体系化下注的标志性时刻。

---

## 二、三种主流形态

### 形态 A：纯软件平台（卖给律师/律所）

- **Harvey**：估值 \$11B，企业级 AI 法务平台
- **Eve**：\$1B+，专攻原告侧（plaintiff）人身伤害律所
- **EvenUp**：\$2B+，专攻个人伤害（PI）证据合成
- **Legora**（前 Leya，YC W24）：欧洲律师 AI 工作台
- **Spellbook / Ironclad / Robin AI**：合同领域

### 形态 B：AI 原生律所（自己就是律所）

- **Crosby**：Sequoia 系，硅谷标杆案例
- **General Legal**：YC W26，前 Casetext 班底重组
- **Eudia Counsel**：在 Arizona ABS 框架下运作，攻 M&A 尽调
- **Moritz**：YC alum，欧洲（挪威）
- **LegalOS**：移民法 AI 原生律所
- 据 Artificial Lawyer 已有 **27+ 家** AI-native 律所目录在追踪

### 形态 C：混合（软件 + 服务外包）

- **EvenUp** 2026.05 推出 **Pre-Litigation-as-a-Service**，从软件商扩展为 PI 律所的运营外包
- 这是值得关注的趋势：**软件公司向下沉到服务层**

---

## 三、关键玩家深度剖析

### 1. Crosby —— Sequoia 钦点的 AI 律所标杆

| 项     | 详情 |
| ------ | ---- |
| 成立   | 2024，Ryan Daniels（Stanford Law）+ John Sarihan（前 Ramp 工程师） |
| 模式   | 注册律所 + 职业责任保险，**通过 Slack 接单交付合同审查**（NDA/MSA/销售合同）|
| SLA    | 合同审查 **< 1 小时**（2025.10 起扩展邮件接单，4 小时 SLA）|
| 客户   | Cursor、Clay、UnifyGTM 等硅谷高速增长公司 |
| 融资   | **累计 \$85.8M**：Seed \$5.8M（2025.06 Sequoia + Bain）→ Series A \$20M（2025.10 Index 领投，Cooley 跟投）→ **Series B \$60M（2026.03 Lux + Index 领投，Sequoia/Elad Gil/01 Advisors 跟投）** |

**Sequoia 的投资逻辑**（来自 Sequoia 官方 partnering 文章）：

- 合同已经成为公司扩张的瓶颈，AI 已经加速了代码/设计/销售，但法律还没动
- **法律工作仍需要人类判断**："people are still better than technology at capturing the subtleties of certain clauses" —— 这是为什么走律所形态而不是 pure software
- 价值主张：**速度 + 成本 + 定制化知识库**（随客户使用而沉淀双方偏好）
- 客户实证：Cursor 等公司销售合同 ≤ 2 小时完成

> **注意：Crosby 不是 YC 公司**，它是 Sequoia 直接孵化路径。

### 2. General Legal —— YC W26，AI 原生律所教科书

| 项         | 详情 |
| ---------- | ---- |
| 成立       | 2026.01.07 正式启动（YC W26 批次）|
| 创始团队   | 三位 Harvard JD：**Ryan Walker**（前 Casetext CTO）、Javed Qadrud-Din（Casetext 第一代语义搜索作者）、JP Mohler（前 Cooley/WilmerHale 律师 + iOS/Android 工程师）—— **本质上是 Casetext AI 团队的二次创业** |
| 目标客户   | 成长期公司（Series B-D，\$100M+ 融资）|
| 定价       | NDA/简短合同 \$250；MSA \$500；APA 等高复杂度更高 |
| SLA        | 合同 1 小时返回 |
| 单位经济   | **MSA 传统律所 8-10 小时 → General Legal 2.2 小时**（消除约 80% 人工）；目标 18 个月内降至 30-40 分钟 |
| 毛利       | **40-50%**（目标向软件级靠拢）|
| 用工模式   | 不招新人，**只招 BigLaw 第 5-8 年级律师或 10 年+ in-house**，作为"主题专家"使用 |
| 业绩       | 3 个月内 110+ 客户、\$1M ARR、累计 **\$11.5M Seed/pre-Seed** |
| 亮点       | 全球**第一家提供 MCP server** 的律所（legalmcp.org）|

这家可能是研究 AI 原生律所运营模式最透明的样本。

### 3. Eudia Counsel —— Arizona ABS 框架下的企业法务 AI 律所

| 项         | 详情 |
| ---------- | ---- |
| 母公司 Eudia | 2023 成立，2025.02 Series A **\$105M**，**General Catalyst** 领投（Floodgate、Sierra Ventures 跟投）|
| 律所形态   | 2025.06 经 Arizona 最高法院批准、按 **Alternative Business Structure（ABS）**规则成立 **Eudia Counsel** |
| 目标       | 企业法务部门、M&A 尽调 |
| 核心理念   | "杀死 billable hour"，提出 **Company Brain** —— 把外部律所流失的机构知识沉淀回甲方 |
| 模式       | AI 增强（"AI-augmented"），不是纯替代律师 |

### 4. Harvey —— 卖软件给 AmLaw 100 的吸金兽

| 时间      | 估值   | 投资方 |
| --------- | ------ | ------ |
| 2025.02   | \$3B   | Sequoia 领投 Series D \$300M |
| 2025.06   | \$5B   | Series E \$300M，**Kleiner Perkins + Coatue** 联合领投 |
| 2025.12   | \$8B   | **a16z** 领投 |
| **2026.03** | **\$11B** | \$200M，**GIC + Sequoia** 联合领投（Sequoia 三度加注）|

Harvey 不是律所，是给 AmLaw 100 律师事务所卖企业级 AI 平台。CEO 公开说接近 **\$100M ARR**。值得注意：Anthropic 自己也开始下场做 legal services（TechCrunch 2026.05 报道）。

### 5. Eve —— 原告律所专用 AI

| 项         | 详情 |
| ---------- | ---- |
| 模式       | 卖给原告（plaintiff）律所的端到端 AI 平台 |
| Series A   | 2025.01 \$47M |
| Series B   | 2025.09 **\$103M @ \$1B+** —— Spark 领投，a16z / Lightspeed / Menlo 跟投 |
| 业绩       | 8 个月新增 350 家律所客户，已 450+ 家，年处理 200K+ 案件，累计帮客户获赔 \$3.5B+ |

### 6. EvenUp —— PI 法律 AI 独角兽 + 服务化先驱

| 项         | 详情 |
| ---------- | ---- |
| 估值       | 2024 Series D \$1B+；**2025.10 Series E \$150M @ \$2B+** |
| 累计融资   | \$385M |
| 业绩       | 周处理 10K 案件、累计 20 万案、\$10B 赔付、覆盖美国 PI 律所 Top 100 中的 20% |
| 关键转向   | **2026.05 推出 Pre-Litigation-as-a-Service** —— 从软件向服务延伸 |

### 7. Casetext —— 已退出的标杆（也是后续创业人才源）

2023 年 Thomson Reuters 以 **\$650M 现金**收购，发布 4 个月就卖。CoCounsel 集成进 TR 全系产品，**2026.02 用户突破 100 万**。Casetext 退出后，原 CTO + 核心成员重新组团做了 General Legal。

### 8. YC 系其他动作

| 公司          | 定位                          | 备注 |
| ------------- | ----------------------------- | ---- |
| **Crimson**   | YC S2025，复杂诉讼 AI         | 帮律师每周省 10 小时 |
| **Lexi**      | YC，公司法 "AI Associate"     | 已处理 135K 文档、7K 案件 |
| **Moritz**    | YC alum，挪威                 | 毕业后 4 天关闭 \$9M pre-seed，做欧洲 AI 律所 |
| **Blueshoe**  | YC                            | 法律推理 AI 平台 |
| **Legora**（前 Leya） | YC W24                | 律师 AI 工作台，15 国 200+ 客户 |

---

## 四、监管创新：AI 原生律所为什么能在美国合法存在？

传统美国律师监管规则（ABA Model Rule 5.4）禁止非律师持股律所、禁止 fee-sharing。这意味着 VC **不能直接投律所**。但近 5 年出现了两个突破口：

### Arizona —— ABS（Alternative Business Structure）路径

- 2020 年起废除 Rule 5.4 限制，**允许非律师持股律所**
- 2024.09 已批准第 100 家 ABS
- **Eudia Counsel 是 2025.06 经 Arizona 最高法院批准的 ABS** —— 这是 VC-backed AI 律所走通监管的关键样本
- Stanford 研究表明：**几乎没看到消费者损害证据**

### Utah —— Regulatory Sandbox 路径

- 创建"沙盒"机制，给特许实体豁免 Rule 5.4 + UPL（未授权执业）
- 允许 nonlawyer providers 和**纯技术服务（含 AI）**直接提供法律服务
- 沙盒至今**仅 20 起消费者投诉**，对低收入人群覆盖效果显著

### Crosby/General Legal 的合规路径

不一定通过 ABS —— 它们走的是**律师持股 + AI 是内部工具**的合规路径，律所本身由持牌律师拥有，AI 不是"非律师股东"。

---

## 五、商业模式 & 单位经济学

VC 投这条赛道的核心 thesis：

> **传统律所是线性人力业务，律师拿走 60%+ 利润，毛利结构差；AI 原生律所能在保持服务质量的前提下，做出软件级的毛利结构。**

| 指标             | 传统律所            | AI 原生律所            |
| ---------------- | ------------------- | ---------------------- |
| MSA 审查耗时     | 8-10 小时           | 2.2 小时（目标 0.5h）  |
| 计费方式         | \$400-1000/h × 小时 | 包干价 \$250-\$500     |
| 毛利率           | 30-40%              | 40-50%（目标软件级 70%+）|
| 边际客户成本     | 高（必须雇人）      | 低（AI 边际成本接近零）|
| 用工策略         | 大量初级律师金字塔  | 只招高级律师，AI 替代初级 |

**关键产品形态共性**：

- Slack-first 接单（Crosby、General Legal）
- 小时级 SLA（vs 传统几天到几周）
- 客户专属知识库随使用沉淀
- 定价从小时费转**包干价/订阅**

---

## 六、市场规模 & 趋势

- **全球 legal tech 市场**：\$29.81B（2025）→ \$65.51B（2034），CAGR 9.14%
- **北美主导**，AmLaw 100 在重金投 AI
- **2026 进入整合期**：2022 年后涌现的几百家 legal AI 创业公司开始洗牌，赢家通吃趋势明显
- **Anthropic 等模型公司下场**：开始直接做 legal services，model layer 公司向应用层挤压
- **VC 押注扩散**：投钱不只投 model layer 公司，更多投垂直应用

---

## 七、风险与挑战

1. **监管不确定**：Crosby 等不在 ABS 框架下的模式，本质依赖"AI 是工具不是律师"这一认定，未来可能被各州 bar 挑战
2. **职业责任保险**：AI 出错谁负责？目前都靠律所背书，模型问题导致的实际损害还没大案例
3. **客户集中风险**：早期客户高度集中在硅谷创业公司（Cursor、Clay）—— 如果硅谷预算收紧，业务波动大
4. **人才悖论**：模式依赖高级律师，但 BigLaw 第 5-8 年级律师同时是最贵、最难挖的群体
5. **模型层挤压**：Anthropic 等下场做应用，工具层创业公司护城河受质疑
6. **服务 vs 软件的估值倍数差**：VC 给 SaaS 倍数高，给服务公司倍数低 —— 这就是为什么 Crosby 要叙述"软件-like 毛利"

---

## 八、关键观察

1. **中国版可行性**：监管路径比美国更难（律师法对非律师股东更严），可能要走"科技公司 + 合作律所"双壳结构
2. **垂直切口比通用更稳**：Eve（原告）、EvenUp（PI）、Eudia（M&A）、LegalOS（移民）都选了细分赛道；通用合同审查赛道已经很拥挤
3. **Slack-first 是当前默认 UX**：硅谷客户在哪 = 产品在哪，传统 portal/邮件已经被淘汰
4. **包干价是关键定价创新**：彻底告别 billable hour，本质是把不可预测变可预测
5. **数据飞轮**：客户专属知识库是真正的护城河，比 model layer 的护城河更稳

---

## 数据来源

1. [Sequoia: Partnering with Crosby](https://sequoiacap.com/article/partnering-with-crosby-a-law-firm-at-the-speed-of-ai/)
2. [Crosby Raises \$20M Series A — Upstarts Media](https://www.upstartsmedia.com/p/crosby-ai-law-firm-raises-20-million)
3. [Crosby Series B \$60M — TAMradar](https://www.tamradar.com/funding-rounds/crosby-series-b-60m)
4. [Artificial Lawyer: How Do AI-Native Law Firms Work?](https://www.artificiallawyer.com/2026/03/31/how-do-ai-native-law-firms-work/)
5. [Harvey \$11B Valuation — TechCrunch](https://techcrunch.com/2026/03/25/harvey-confirms-11b-valuation-sequoia-triples-down/)
6. [Harvey \$11B — CNBC](https://www.cnbc.com/2026/03/25/legal-ai-startup-harvey-raises-200-million-at-11-billion-valuation.html)
7. [Eve \$103M Series B @ \$1B](https://www.lawnext.com/2025/09/eve-ai-driven-platform-for-plaintiff-side-law-firms-raises-103-million-in-series-b-round.html)
8. [EvenUp Series E @ \$2B — Fortune](https://fortune.com/2025/10/07/exclusive-evenup-raises-150-million-series-e-at-2-billion-valuation-as-ai-reshapes-personal-injury-law/)
9. [EvenUp PLAAS Launch — LawSites](https://www.lawnext.com/2026/05/evenup-extends-beyond-software-with-launch-of-pre-litigation-as-a-service-offering-for-pi-law-firms.html)
10. [Eudia Counsel Launch — Artificial Lawyer](https://www.artificiallawyer.com/2025/09/03/eudia-opens-ai-augmented-law-firm-for-ma/)
11. [Eudia kills the billable hour — Fortune](https://fortune.com/2025/09/03/eudia-legal-tech-ai-startup-killing-billable-hour/)
12. [Stanford: 5 Years of AZ/UT Regulatory Reform](https://law.stanford.edu/2025/06/02/regulatory-innovation-at-the-crossroads-five-years-of-data-on-entity-regulation-reform-in-arizona-and-utah/)
13. [YC General Legal Profile](https://www.ycombinator.com/companies/general-legal)
14. [Moritz YC alum \$9M — Sifted](https://sifted.eu/articles/y-combinator-alum-moritz-raises-9m)
15. [Legora (formerly Leya) — Crunchbase](https://www.crunchbase.com/organization/leya-876d)
16. [Anthropic enters legal services — TechCrunch](https://techcrunch.com/2026/05/12/the-ai-legal-services-industry-is-heating-up-anthropic-is-getting-in-on-the-action/)
17. [IBA: AI-native law firm regulatory innovation](https://www.ibanet.org/AI-native-law-firm-regulatory-innovation-and-fundamental-restructuring-of-legal-service-delivery)
