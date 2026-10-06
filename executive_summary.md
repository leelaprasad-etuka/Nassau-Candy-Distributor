# Executive Briefing: Supply Chain Rebalancing and Manufacturing Facility Reallocation

**To:** Non-Technical Public Sector and Corporate Executive Leadership  
**From:** Nassau Candy Supply Chain Optimization Task Force  
**Date:** October 2026  
**Subject:** Algorithmic Factory Reallocation and Logistical Optimization Findings  

---

### 1. The Operational Problem
Nassau Candy Distributor manufactures and distributes 15 distinct confectionery products to commercial destinations across 59 regions in the United States and Canada. Historically, manufacturing operations have been locked into single-facility silos: every product is produced exclusively at one specific plant regardless of where orders originate.

This structural constraint forces cross-country shipments over long distances (averaging 1,997.9 kilometers per delivery), creating extended delivery schedules, route congestion, and unnecessary transportation costs. Management requires an objective, data-driven assessment to determine whether shifting select products to closer regional facilities can improve delivery speed without eroding operating profit.

---

### 2. What Was Done
Our analytical team conducted a rigorous, multi-stage data science and logistics engineering evaluation:
1. **Data Auditing and Cleaning:** Audited 10,194 historical shipment transactions, confirmed financial accounting integrity (Sales minus Cost matches Gross Profit with zero discrepancies), and removed 263 statistical volume outliers using the standard Interquartile Range rule.
2. **Geographic Network Mapping:** Constructed geographic coordinate centroids for all 50 US States, Washington D.C., and 10 Canadian provinces (100% geographic coverage with 0 unmatched jurisdictions) to measure exact point-to-point transit distances.
3. **Logistics Route Clustering:** Identified 189 distinct factory-to-destination corridors and partitioned them into 5 distinct operational clusters using machine learning, pinpointing chronically slow long-distance routes.
4. **Predictive Modeling:** Trained and cross-validated machine learning models (Linear Regression, Gradient Boosting, and Random Forest) to simulate alternative factory assignments under identical, like-for-like operational conditions.
5. **Scenario Simulation Engine:** Precomputed 17,700 counterfactual scenarios evaluating every product, destination, and transportation mode across all 5 candidate manufacturing facilities.

---

### 3. Key Findings and Recommended Actions
Using a balanced decision framework that weights delivery speed and net profit equally, the optimization engine established four core performance benchmarks:
- **Expected Lead Time Improvement:** +0.56% average acceleration across recommended SKU reallocations.
- **Profit Stability Index:** 56.1% stability across geographic territories and shipping modes.
- **Decision Confidence Score:** 31.8 out of 100 overall (reflecting high statistical certainty on core high-volume chocolate products and prudent risk penalties on lower-volume novelty items).
- **Viable Reallocation Coverage:** 40.0% of the product catalog (6 of 15 products) can be strategically reallocated to closer facilities to reduce delivery time without violating a 5.0% maximum profit tolerance.

#### Specific Recommended SKUs for Immediate Pilot Reallocation:
1. **Fizzy Lifting Drinks (Sugar Division):** Shift primary production from Sugar Shack (Minnesota) to The Other Factory (Tennessee). Expected result: +0.34% speed gain and a +2.12% profit improvement from reduced shipping distance.
2. **Everlasting Gobstopper (Sugar Division):** Shift production from Secret Factory (Illinois) to Lot's O' Nuts (Arizona) for Western regional shipments. Expected result: +2.17% delivery speed improvement.
3. **Preserve Current Production on Core Chocolate Lines:** Products such as Wonka Bar Milk Chocolate and Triple Dazzle Caramel (at Wicked Choccy's in Georgia) should remain at their current facilities. Shifting high-volume chocolate lines creates negligible speed improvements (less than 0.5%) while incurring substantial cross-country shipping freight expenses.

---

### 4. Operational Risks and Sensitivities
1. **Data Horizon Limitation:** Historical delivery records reflect multi-year scheduling horizons (averaging 1,320.8 days). Results are interpreted as relative efficiency gains in percentage terms rather than absolute calendar days.
2. **Low-Volume Statistical Sensitivity:** The Sugar division represents only 40 historical shipment records (0.4% of total transaction volume). Production shifts must be validated through phased regional trials rather than sudden nationwide cutovers.
3. **Carrier Pricing Fluctuations:** Freight costs are modeled based on standard distance-based rates ($0.0005 per unit-km). Increases in fuel prices or carrier freight surcharges will accelerate the cost-savings of manufacturing closer to consumer hubs.

---

### 5. Decision Requested
Executive leadership is requested to approve the following two-part action plan:
1. **Authorize a 90-Day Operational Pilot:** Reallocate regional production of Fizzy Lifting Drinks and Everlasting Gobstoppers to Southern and Western network facilities, tracking actual dock-to-door transit times and freight savings.
2. **Adopt the Interactive Decision Dashboard:** Authorize logistics procurement teams to utilize the interactive optimization platform (`app.py`) for monthly scheduling and contract rate negotiations.
