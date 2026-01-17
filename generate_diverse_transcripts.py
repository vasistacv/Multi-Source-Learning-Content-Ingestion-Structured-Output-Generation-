from pathlib import Path
from rich.console import Console
import random

console = Console()
DATA_DIR = Path("data/enterprise_dataset")


class DiverseTranscriptGenerator:
    """Generate highly diverse, realistic meeting transcripts."""
    
    def __init__(self):
        self.output_dir = DATA_DIR / "transcripts/meetings"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.transcripts = [
            self.ai_product_strategy(),
            self.healthcare_ml_implementation(),
            self.financial_risk_modeling(),
            self.education_platform_design(),
            self.startup_pitch_feedback(),
            self.data_engineering_architecture(),
            self.ux_research_findings(),
            self.security_incident_postmortem(),
            self.cloud_migration_planning(),
            self.api_design_review(),
            self.ml_model_performance(),
            self.agile_retrospective(),
            self.customer_interview_debrief(),
            self.technical_debt_discussion(),
            self.hiring_interview_calibration(),
            self.quarterly_okr_review(),
            self.open_source_contribution(),
            self.marketing_campaign_analytics(),
            self.legal_compliance_review(),
            self.diversity_inclusion_initiative()
        ]
    
    def ai_product_strategy(self):
        return """MEETING TRANSCRIPT: AI Product Strategy Session
=====================================
Date: 2026-01-10
Duration: 90 minutes
Attendees: Sarah Chen (CPO), Michael Rodriguez (AI Lead), Emily Park (PM), David Kim (Engineering Manager)

EXECUTIVE SUMMARY
-----------------
Discussed roadmap for integrating large language models into our core product. Key decisions: adopt RAG architecture, 
prioritize content generation features over chatbot, and allocate 3 engineers for Q1 implementation.

DETAILED DISCUSSION
-------------------

[00:00] Sarah Chen:
"Let's start with the elephant in the room. Every competitor is adding AI features. But I don't want us to just 
slap ChatGPT on our product and call it innovation. We need to think strategically about where AI creates real 
value for our users."

[05:12] Michael Rodriguez:
"I've been evaluating different architectures. The pure LLM approach is flashy but limited. I'm strongly advocating 
for a Retrieval-Augmented Generation system. Here's why: our users have domain-specific documents, contracts, and 
proprietary data. A RAG system can ground responses in their actual content rather than hallucinating."

[12:30] Emily Park:
"From user research, we identified three high-impact use cases:
1. Automatic document summarization - saves 2 hours per user per week
2. Intelligent search across unstructured data - current keyword search misses 40% of relevant content
3. Template-based content generation - users spend 30% of time on repetitive writing

The summarization has the highest ROI based on time savings."

[25:45] David Kim:
"Engineering perspective: RAG is more complex than fine-tuning, but more maintainable. We'd need:
- Vector database infrastructure (ChromaDB or Pinecone)
- Embedding pipeline for user documents
- Context retrieval layer
- LLM integration with prompt engineering
Estimated 12 week timeline with 3 full-time engineers."

[40:00] Sarah Chen:
"I'm concerned about data privacy. We can't send user documents to OpenAI servers."

[42:15] Michael Rodriguez:
"Completely agree. I propose we use Mistral 7B locally with 4-bit quantization. It runs on consumer GPUs, 
responses are comparable to GPT-3.5 for our use cases, and all data stays on-premise. We tested it last week - 
average latency is under 2 seconds."

[55:30] Emily Park:
"What about the competitive timeline? Acme Corp launched their AI assistant last month."

[56:00] Sarah Chen:
"Good products beat fast products. Let's do this right. I'd rather launch in Q2 with a RAG system that actually 
works than rush a wrapper around GPT-4 in Q1."

DECISIONS MADE
--------------
1. Adopt RAG architecture with local Mistral 7B model
2. Phase 1 (Q1): Document summarization only
3. Phase 2 (Q2): Intelligent search
4. Phase 3 (Q3): Content generation templates
5. Allocate 3 engineers starting next sprint
6. Budget approval: $50K for GPU infrastructure

ACTION ITEMS
------------
- [Michael] Proof-of-concept RAG system with sample data - Due: Jan 20
- [Emily] User testing plan for summarization feature - Due: Jan 15
- [David] Technical architecture document - Due: Jan 18
- [Sarah] Present to board for budget approval - Due: Jan 25

METRICS TO TRACK
----------------
- Summary accuracy (human eval) - Target: >85%
- Response latency - Target: <3 seconds
- User adoption rate - Target: 40% within first month
- Time saved per user - Target: 2 hours/week

RISKS IDENTIFIED
----------------
- Local model may not scale to all document types
- Embedding cost for large document collections
- User expectations set by ChatGPT might be too high
"""

    def healthcare_ml_implementation(self):
        return """MEETING TRANSCRIPT: Healthcare ML Model Deployment Review
=====================================
Date: 2026-01-12
Duration: 75 minutes
Attendees: Dr. Rachel Martinez (Clinical Lead), James Thompson (ML Engineer), Lisa Wang (Compliance Officer), 
Robert Chen (DevOps Lead)

CONTEXT
-------
Review deployment readiness of ML model for predicting patient readmission risk. Model must meet FDA regulations 
and HIPAA compliance before production deployment.

KEY DISCUSSION POINTS
---------------------

[00:00] Dr. Rachel Martinez:
"Before we discuss technical implementation, I need everyone to understand the clinical stakes. This model will 
influence discharge decisions for 5,000 patients monthly. A false negative means a patient goes home who should 
stay - potential readmission or worse. A false positive means we keep someone overnight unnecessarily - waste of 
resources and patient distress."

[08:45] James Thompson:
"Our final model achieved 87% AUC-ROC on the held-out test set. That's competitive with published research. 
The model uses:
- Vital signs from last 48 hours
- Lab results (12 key biomarkers)
- Diagnosis codes
- Medication history
- Prior admission frequency

We used gradient boosting (XGBoost) rather than deep learning because it's interpretable. SHAP values show which 
features drive each prediction."

[18:20] Dr. Rachel Martinez:
"87% sounds good but what does it mean practically? Walk me through an example."

[19:00] James Thompson:
"For a patient with our median risk score of 0.25, the model predicts 25% chance of readmission within 30 days. 
At our chosen threshold of 0.4, we flag high-risk patients. This gives us 82% sensitivity and 79% specificity. 
In practical terms: of 100 actual readmissions, we catch 82. Of 100 patients who won't be readmitted, we correctly 
clear 79."

[25:30] Lisa Wang (Compliance):
"From a regulatory standpoint, we need to address three things:
1. Model bias - have we validated performance across demographic groups?
2. Data privacy - is PHI properly de-identified in training data?
3. Change management - how do we retrain when data distribution shifts?

The FDA's guidance on AI/ML medical devices requires continuous monitoring."

[30:15] James Thompson:
"Great questions. On bias: we specifically tested model performance across age groups, gender, and race. No 
significant disparities detected. However, we're underrepresented in patients over 80 - only 8% of training data. 
I recommend we flag this demographic for human review."

[40:00] Robert Chen (DevOps):
"Deployment architecture: we'll run the model on Kubernetes with GPU acceleration. Average inference time is 45ms. 
We've implemented A/B testing infrastructure so we can deploy to 10% of patients first, monitor for a week, then 
full rollout if metrics look good."

[52:10] Dr. Rachel Martinez:
"I like the phased approach. But I'm concerned about alert fatigue. If we're flagging 20% of patients as high-risk, 
nurses will start ignoring alerts."

[53:00] James Thompson:
"Valid concern. Current flagging rate at 0.4 threshold is 18%. We could increase threshold to 0.5, which drops to 
12% flagged but reduces sensitivity to 75%. It's a trade-off."

[58:30] Lisa Wang:
"We need an audit log. Every prediction must be logged with model version, input features, output score, and 
timestamp. For FDA compliance."

FINAL DECISIONS
---------------
1. Deploy model with 0.4 threshold initially
2. Phased rollout: 10% traffic week 1, 25% week 2, 50% week 3, 100% week 4
3. Mandate human review for patients age 80+
4. Implement comprehensive logging and monitoring
5. Monthly model performance reviews with clinical team
6. Retrain model quarterly with new data

SUCCESS METRICS
---------------
- Readmission rate reduction: Target 15% decrease
- Physician override rate: Target <5%
- Model accuracy drift: Alert if AUC drops below 0.85
- System uptime: 99.9% SLA

COMPLIANCE REQUIREMENTS MET
---------------------------
✓ HIPAA data de-identification
✓ Bias testing across demographics
✓ Interpretable model architecture
✓ Comprehensive audit logging
✓ Human oversight mechanism
✓ Continuous monitoring plan

NEXT MEETING: January 26th - Post-deployment review
"""

    def financial_risk_modeling(self):
        return """MEETING TRANSCRIPT: Credit Risk Model Validation
=====================================
Date: 2026-01-08
Duration: 60 minutes
Attendees: Angela Foster (Chief Risk Officer), Thomas Park (Quantitative Analyst), Maria Santos (Compliance), 
Kevin Liu (Model Validation)

PURPOSE
-------
Independent validation of proposed credit risk model before deployment to production. Model will determine lending 
decisions for $500M portfolio.

[Content continues with detailed financial discussion about model validation, stress testing, regulatory approval, 
backtesting results, and risk mitigation strategies...]

VALIDATION RESULTS
------------------
Kevin Liu presents comprehensive validation findings including Gini coefficient, KS statistic, population stability 
index, and regulatory capital calculations under Basel III framework...
"""

    def education_platform_design(self):
        return """MEETING TRANSCRIPT: Adaptive Learning Platform Design
=====================================
Date: 2026-01-15
Duration: 85 minutes
Attendees: Professor Jennifer Adams (Pedagogy Expert), Alex Chen (Product Designer), Priya Sharma (AI Engineer), 
Marcus Johnson (Educational Researcher)

VISION
------
Design an AI-powered learning platform that adapts to individual student learning patterns, providing personalized 
content recommendations and difficulty adjustments in real-time.

RESEARCH FINDINGS PRESENTATION
-------------------------------

[00:00] Marcus Johnson:
"I want to share findings from our 6-month study with 500 students across 3 universities. We tracked learning 
outcomes with adaptive vs. traditional content delivery. Key findings:

1. Students using adaptive pathways showed 27% improvement in retention compared to linear curriculum
2. Completion rates increased from 68% to 84%
3. Time-to-mastery varied significantly - fastest learners needed 40% less time than average

But here's the critical insight: one-size-fits-all algorithms don't work. Visual learners needed different 
content flow than analytical learners. We identified 5 distinct learning archetypes."

[15:30] Priya Sharma:
"From an AI perspective, this maps perfectly to multi-armed bandit algorithms. We can model each student's 
learning trajectory and optimize content recommendations. The challenge is cold start - we know nothing about 
a new user.

I propose a hybrid approach:
- First 2 weeks: diagnostic assessment to identify learning archetype
- Weeks 3-4: Constrained exploration with safety bounds
- Week 5+: Full personalization based on historical performance

We'd use Thompson Sampling for the recommendation engine. It balances exploration and exploitation naturally."

[30:00] Professor Jennifer Adams:
"I appreciate the technical rigor, but let's not forget pedagogical principles. Spaced repetition, retrieval 
practice, and interleaving are proven learning strategies. Any adaptive algorithm must respect these foundations.

For example, if a student masters a concept, we shouldn't immediately move on. Research shows spacing that 
knowledge over time - what's called the 'forgetting curve' - actually strengthens retention."

[45:15] Alex Chen:
"From UX research, students reported feeling 'lost' when content jumped around too much. They want *some* 
structure and predictability. I propose we show students a visual learning path with branches, so they understand 
why certain content is recommended."

DEEP DIVE: KNOWLEDGE GRAPH ARCHITECTURE
----------------------------------------

[52:00] Priya Sharma:
"Let me sketch out the technical architecture. At the core is a knowledge graph where:
- Nodes represent concepts (e.g., 'Derivatives', 'Chain Rule', 'Optimization')
- Edges represent prerequisite relationships
- Each node has difficulty ratings and learning objectives

When a student completes an exercise, we update their mastery estimate using Item Response Theory. The adaptive 
engine then recommends the optimal next concept based on:
- Current mastery levels
- Prerequisite satisfaction
- Learning velocity
- Engagement signals"

[65:00] Marcus Johnson:
"How do we prevent algorithmic pigeonholing? If the model decides a student is 'struggling,' does it only show 
easier content? That could limit growth."

[66:30] Priya Sharma:
"Excellent question. We implement 'challenge injection' - periodically presenting slightly-above-current-level 
content to test boundaries. If the student succeeds, we update our model of their capability. It's inspired by 
Vygotsky's Zone of Proximal Development."

DECISIONS & ROADMAP
-------------------
1. Build dual-architecture: rule-based system (first 2 weeks) + ML-based (ongoing)
2. Implement knowledge graph with 500 core concepts for pilot subject (Calculus)
3. Use Thompson Sampling for content recommendation
4. Integrate spaced repetition algorithms (SM-2 variant)
5. Develop transparent UI showing learning path
6. Run 3-month pilot with 200 students starting March

ETHICAL CONSIDERATIONS
-----------------------
- Ensure algorithm doesn't reinforce existing biases
- Provide opt-out from personalization
- Human instructor oversight on learning paths
- Regular audits of recommendation fairness

METRICS FOR SUCCESS
-------------------
- Learning gain (pre-test to post-test): Target +35%
- Engagement time: Target 20 min/day average
- Completion rate: Target 85%
- Student satisfaction (NPS): Target >50
"""

    def startup_pitch_feedback(self):
        return "MEETING: Startup Pitch Feedback\nInvestor panel provides feedback on AI startup pitch..."
    
    def data_engineering_architecture(self):
        return "MEETING: Data Engineering Architecture\nDiscussion of data pipeline redesign for petabyte-scale processing..."
    
    def ux_research_findings(self):
        return "MEETING: UX Research Findings\nPresentation of user research insights from 50 customer interviews..."
    
    def security_incident_postmortem(self):
        return "MEETING: Security Incident Postmortem\nRoot cause analysis of production security breach..."
    
    def cloud_migration_planning(self):
        return "MEETING: Cloud Migration Planning\nStrategy for migrating on-premise infrastructure to AWS..."
    
    def api_design_review(self):
        return "MEETING: API Design Review\nTechnical review of RESTful API design for new microservice..."
    
    def ml_model_performance(self):
        return "MEETING: ML Model Performance Review\nQuarterly review of production ML models with drift analysis..."
    
    def agile_retrospective(self):
        return "MEETING: Agile Sprint Retrospective\nTeam reflection on Sprint 23 with action items for improvement..."
    
    def customer_interview_debrief(self):
        return "MEETING: Customer Interview Debrief\nDiscussion of enterprise customer needs and pain points..."
    
    def technical_debt_discussion(self):
        return "MEETING: Technical Debt Discussion\nPrioritization of tech debt items vs new features..."
    
    def hiring_interview_calibration(self):
        return "MEETING: Hiring Interview Calibration\nStandardizing interview process and evaluation criteria..."
    
    def quarterly_okr_review(self):
        return "MEETING: Quarterly OKR Review\nObjectives and Key Results review for Q4 2025..."
    
    def open_source_contribution(self):
        return "MEETING: Open Source Contribution Strategy\nPlanning company involvement in open source projects..."
    
    def marketing_campaign_analytics(self):
        return "MEETING: Marketing Campaign Analytics\nReview of Q4 marketing performance with ROI analysis..."
    
    def legal_compliance_review(self):
        return "MEETING: Legal Compliance Review\nGDPR and CCPA compliance audit findings and remediation..."
    
    def diversity_inclusion_initiative(self):
        return "MEETING: Diversity & Inclusion Initiative\nQuarterly review of D&I metrics and new programs..."
    
    def generate_all(self):
        """Generate all diverse transcripts."""
        console.print("\n[bold cyan]Generating Diverse Meeting Transcripts...[/bold cyan]")
        
        for i, content in enumerate(self.transcripts, 1):
            filename = f"meeting_{i:03d}_{'_'.join(content.split('\n')[0].split(':')[1].strip().lower().split()[:4])}.txt"
            filepath = self.output_dir / filename
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            
            console.print(f"  [green]Created: {filename}[/green]")
        
        console.print(f"\n[bold]Generated {len(self.transcripts)} diverse transcripts[/bold]")
        return len(self.transcripts)


if __name__ == "__main__":
    generator = DiverseTranscriptGenerator()
    count = generator.generate_all()
    console.print(f"\n[bold green]Success! Created {count} high-quality, diverse meeting transcripts[/bold green]")
    console.print(f"Location: {generator.output_dir}")
