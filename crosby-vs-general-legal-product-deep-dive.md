---
title: "Crosby vs General Legal 产品深度拆解"
subtitle: "AI 原生律所赛道两大代表性玩家对比"
date: "2026 年 5 月"
---

# Crosby vs General Legal —— 产品深度拆解

> 本文为《AI 原生律所赛道调研报告》的姊妹篇，专门拆解美国 AI 原生律所赛道两个最受关注的玩家。

---

## 一、Crosby 产品拆解

### 团队 & 规模

| 项 | 详情 |
| --- | --- |
| 律师数 | **约 2 打**（24 人左右），律师与工程师同处一办公室 |
| 律师薪酬 | 现金低于 BigLaw，但**单律师每小时合同处理量显著更高**（杠杆换薪酬）|
| 律师角色 | 双重身份 —— 做法律审查 + **做 data labelers**，标注 AI 边界、训练模型 |
| 创始人 | **Ryan Daniels**（Stanford Law、Cooley 律师 → A.Team in-house → BCV EIR） + **John Sarihan**（前 Ramp 工程 Tech Lead，做监管/牌照系统）|
| 相识场合 | 两人在 Bain Capital Ventures 认识（Ryan 是 EIR）|

### 产品形态

| 项 | 详情 |
| --- | --- |
| 内部平台名 | **"Bailiff"** —— 接单 + 路由 + 优先级判定 |
| 接入渠道 | **Slack + 邮件 + CLM 触发器**（直接对接客户的 Ironclad/Ariba 等）|
| 中位响应时间 | **58 分钟** |
| 处理量 | 约 **1000 合同 / 3 周**（约 333/周，约 47/天）|
| 合同类型 | NDAs、MSAs、DPAs、vendor agreements |
| 工作流 | 客户 → Bailiff → AI 自动优先级 + 路由律师 → AI 标红 + 例外 → 律师只看复杂点 |

### 定价

> "Upfront Pricing" 固定费率，**不按小时**。具体数字未公开（推测企业订阅或合同包），客户实证："12 小时完成传统大所 6 周的工作量"。

### 客户名单（已公开）

Cursor、Clay、Unify、Cartesia、Alloy、Overjet、Rogo、Tishman Speyer、Gumloop —— **几乎都是 a16z/Sequoia/Index 系硅谷高速增长 B 轮+ 公司**。

### 投资人 BCV 的核心论点

- **TAM**：\$18B 死区（白鞋大所 \$1000/h vs 凑合用 ChatGPT 之间）
- **合同量级跳跃**：成长期公司从 10 MSA/月 → 50+/月 是常态
- **数据壁垒**："contracts as code"，反复谈判产生**条款级数据飞轮**
- **关键判断**：纯 SaaS legal tech 不能赢，因为"如果服务 AmLaw 律所，就跟客户产生利益冲突"

### 即将推出

- **跨团队路由**：自动把合同里的安全条款发安全团队、营销条款发营销团队
- **跨客户 benchmark**：基于历史数据预测对手方的反应（"对方大概率会拒这条"）

---

## 二、General Legal 产品拆解

### 团队 & 规模

| 项 | 详情 |
| --- | --- |
| 创始人三人组 | **Ryan Walker**（前 Casetext CTO、被 TR 收购后跑 CoCounsel VP）、**Javed Qadrud-Din**（前 Casetext 第一代语义搜索作者、TR Director of ML）、**JP Mohler**（前 WilmerHale/Cooley 律师 + TR 高级 ML 研究员）|
| 团队来历 | **整个 Casetext 核心 AI 班底重组** |
| 律师团 | "US-barred attorneys"，具体规模未披露 |
| 客户数 | **250+** 成长期公司 |

### 产品形态

| 项 | 详情 |
| --- | --- |
| 接入渠道 | **Slack + 邮件 + web portal + MCP server** —— 业内首个生产级 MCP |
| SLA | 标准合同**几小时**，3 小时初响应保证（**仅工作日 5AM-6PM PT**）|
| 合同类型 | 商业、雇佣、公司注册、早期融资（SAFE 等）、定制起草 |
| 工作流 | AI 分类器识别条款 → 客户专属 playbook 风险评估 → 律师复核 → 版本追踪 |

### MCP Server —— 独家产品差异化

这是 General Legal 最有意思的产品创新：**第一家把自己包装成 AI agent 可调用工具的律所**。

**API 端点**：`https://mcp.general.legal/mcp`

**鉴权**：OAuth 2.1（自动发现，**无需 API key**）

**暴露的 4 个 tool**：

| Tool | 功能 |
| --- | --- |
| `upload_contract` | 提交 .docx，返回 signed URL |
| `confirm_upload` | 触发审查流水线，可附上下文 |
| `list_contracts` | 查询状态：`ai_review` → `attorney_queue` → `attorney_review` → `delivered` → `completed` |
| `download_contract` | 下载带 redlines + 评论的律师审查版 |

**使用方式**：Claude Code 一行命令安装、Claude Desktop JSON 配置、任何支持 Streamable HTTP 的客户端均可接。**整个交互在用户与 AI 助手的自然对话里完成** —— "嘿 Claude，把这份 MSA 发给 General Legal 审一下" 即可。

**战略意义**：在 Agent 经济里，General Legal 把自己定位为 **agent 的法律 API**，而不是给人用的产品。

### 定价（完全公开）

| 类型 | 价格 |
| --- | --- |
| 短合同（≤3 页） | **\$250** |
| 标准合同（3-50 页） | **\$500** |
| 长合同（50+ 页） | **\$10/页** |
| 完整谈判版 | **\$1000**（含无限轮次和电话）|
| 定制起草 | **\$2000** |

**flat fee 覆盖到签约为止的全部往返，包括跟对方谈判**。

### 客户名单（已公开）

Celltype、LightAnchor、Cardboard、Clair Health、Pollinate、Opendate、Sammy Labs、Booko —— **YC + 早期 Series A 偏多**。

### 关键内部数据

- **AI 处理约 80% 工作量**，律师做 20% 的判断/复核
- **每律师杠杆比 5-10x**（vs 传统 1x 律师 = 1x 输出）
- MSA 律师耗时：2.2 小时（目标 18 个月降到 30-40 分钟）

---

## 三、关键对比矩阵

| 维度 | Crosby | General Legal |
| --- | --- | --- |
| **VC 路径** | Sequoia 系（非 YC） | YC W26 |
| **创始 DNA** | 律师 + Ramp 工程 | Casetext AI 班底 + Cooley 律师 |
| **客户画像** | 硅谷 Series B+ 红人公司（Cursor、Clay） | YC + 早期 Series A 偏多 |
| **接入** | Slack + 邮件 + CLM 集成 | Slack + 邮件 + web + **MCP server** |
| **agent 友好度** | 中（CLM 集成） | **高（原生 MCP）** |
| **定价透明度** | 不公开 | **完全公开** |
| **SLA** | 中位 58 分钟，24/7 | 几小时，工作日 5AM-6PM PT |
| **核心壁垒** | 内部 Bailiff 平台 + 客户数据沉淀 | MCP-first 战略 + Casetext IP/经验 |
| **律师团** | 约 24 人，BigLaw 各级混合 | 未披露规模，US-barred 律师 |
| **AI/人分工** | AI 标红 + 例外，律师收尾 | AI 80% + 律师 20% 复核 |
| **未来扩展** | 跨团队路由 + benchmark 数据 | 持续做 MCP 工具集合 |
| **客户实证** | "12h 完成传统 6 周" | "MSA 8-10h → 2.2h" |
| **融资** | 累计 \$85.8M，Series B \$60M @ Lux | 累计 \$11.5M Seed/pre-Seed |
| **估值阶段** | Series B（独角兽附近） | Seed |

---

## 四、战略差异（最关键的一点）

### Crosby = 押注"销售 + 合同瓶颈"

- **Slogan**：*"Built By Lawyers, **For Sales Teams**, To Speed Up Time to Signature"*
- **痛点**：销售合同卡在 legal review → 错失 deal
- **进入点**：CRO / Head of Sales 而非 GC
- **数据飞轮**：每个客户的合同条款偏好深度沉淀
- **主战场**：销售合同流水线

### General Legal = 押注"agent + 法律 API"

- **Slogan**：*"AI native law firm for fast growing startups"*
- **痛点**：创始人自己 ChatGPT review 风险大、找大所太贵
- **进入点**：CEO / CTO / Founder
- **数据飞轮**：MCP 工具被 agent 调用 → 形成 agent 时代的法律基础设施
- **主战场**：AI 时代律所重定义

### 总结判断

**短期客户不太重叠**：
- Crosby 服务 \$50M+ ARR 的销售型公司
- General Legal 服务 \$10M-\$50M ARR 的早期公司

**中期会正面竞争**：两家都会向上和向下扩张，2027 年大概率会撞车。

---

## 五、对中国/国际市场的借鉴

### 可直接迁移的设计

1. **Slack-first / IM-first 接单**：放弃传统 portal 和邮件附件流转 —— 国内对应飞书/钉钉
2. **包干价 + 价格表完全公开**：定价透明是吸引创业公司的关键，传统律所最讨厌这点
3. **小时级 SLA**：把合同审查从"按天"变成"按小时"
4. **客户专属 playbook**：每个客户的偏好沉淀成知识库，是真正的护城河

### 不能直接抄的设计

1. **MCP server 国内还太早**：但飞书 OpenAPI / 钉钉机器人 API 可以做对应
2. **律师团结构**：国内招"BigLaw 5-8 年级律师"成本结构和美国不同，需要重新设计杠杆模型
3. **职业责任保险**：国内律师保险体系不一样，AI 出错的赔付责任不清晰

### 国内做这条赛道的核心难点

| 难点 | 美国情况 | 中国情况 |
| --- | --- | --- |
| VC 持股律所 | Arizona ABS / Utah sandbox 已通 | **明确禁止**，需双壳结构 |
| 律师收入 | BigLaw 50w-100w 美元 | 大所 30-80w 人民币 |
| 软件层付费意愿 | 律师 / 法务付费习惯成熟 | 弱，更习惯按案件付费 |
| 数据合规 | 客户合同保密但能用于训练 | 涉及 PII、商业秘密，训练受限 |

---

## 数据来源

1. [Crosby 官网](https://crosby.ai/)
2. [General Legal 官网](https://general.legal/)
3. [General Legal MCP server 文档](https://legalmcp.org/)
4. [Upstarts Media: Crosby 'AI Law Firm' Raises $20M](https://www.upstartsmedia.com/p/crosby-ai-law-firm-raises-20-million)
5. [BCV: Crosby Redefines Legal Work with AI Contract Automation](https://baincapitalventures.com/insight/crosby-is-redefining-legal-work-with-ai-powered-contract-automation/)
6. [Sequoia: Partnering with Crosby](https://sequoiacap.com/article/partnering-with-crosby-a-law-firm-at-the-speed-of-ai/)
7. [YC General Legal Profile](https://www.ycombinator.com/companies/general-legal)
8. [Artificial Lawyer: How Do AI-Native Law Firms Work?](https://www.artificiallawyer.com/2026/03/31/how-do-ai-native-law-firms-work/)
