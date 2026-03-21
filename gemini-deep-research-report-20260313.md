# **Advanced Methodologies and Architectural Roadmaps for Predicting the 10th Place Finisher in Formula 1**

## **Introduction**

Predicting the precise finishing order of a Formula 1 race is a notoriously complex problem characterized by high variance, stochastic events, and intricate multi-agent interactions. The challenge becomes exponentially more difficult when targeting the 10th place finisher. In the Formula 1 sporting framework, the 10th position serves as the critical demarcation line between scoring championship points and leaving empty-handed.1 The psychological and financial stakes of the midfield battle mean that the strategic approach of a team running in 10th place differs fundamentally from a team running in 11th place.2 Statistical analyses of nearly two decades of historical race data, encompassing 7,800 driver-weekend observations, indicate that the transition across the P10/P11 threshold presents the steepest difficulty gradient in the midfield.3 Ordinal logistic regression models evaluating the probability of positional transitions demonstrate that the P10/P11 threshold holds the most negative threshold value (![][image1]), indicating that breaking into the top 10 requires overcoming a statistically significant performance and strategic barrier.3 Drivers in 11th place must assume significantly higher tactical risks—such as aggressive undercuts or extreme tire offset strategies—to breach the points, while the driver in 10th place adopts highly defensive, risk-averse postures.3

An examination of the current state of the predictive pipeline (versions up to v4.03) reveals a sophisticated and highly developed foundational architecture.4 The deliberate transition from binary classification models to a multi-class Expected Value (EV) architecture mapped across all 20 finishing positions marks a significant conceptual advancement.4 The expansion to 44 distinct features—incorporating track-specific Virtual Safety Car (VSC) rates, pit stop averages, collision rates, and driver-specific Did Not Finish (DNF) recovery metrics—provides a robust statistical baseline.4 The current system effectively utilizes a weighted ensemble of regression and classification models, including Random Forest, LightGBM, XGBoost, and Ridge Regression, adapting the ensemble weights dynamically based on the stage of the season (Early, Mid, and Late season weightings).4

However, predicting midfield outcomes requires continuous, aggressive refinement. The inherent limitation of using pointwise regression and classification models is that they treat each driver's finishing position independently, failing to capture the zero-sum, relative nature of a motor race.5 Furthermore, optimizing for a specific ordinal rank via multi-class expected value calculations exposes underlying vulnerabilities in probability calibration. Compounding these algorithmic challenges is the impending 2026 regulatory overhaul. Formula 1 is entering an era defined by a 50/50 internal combustion and electrical power split, active aerodynamics, and the removal of the traditional Drag Reduction System (DRS).6 Historical continuity will be fractured, requiring models to adapt to entirely new aerodynamic and powertrain physics.9 The ensuing analysis explores advanced statistical approaches, bespoke feature engineering tailored for the midfield, and a comprehensive development plan designed to maximize the predictive accuracy of the 10th-place algorithmic pipeline prior to the race start.

## **Algorithmic Paradigm Shifts for Ordinal Ranking**

The current predictive pipeline relies on a blend of regressors predicting a continuous finishing position and multi-class classifiers predicting the probability distribution of a driver finishing in positions 1 through 20\.4 The predicted driver for P10 is selected based on the highest Expected Value (EV) across the position distribution.4 While this ensemble approach effectively mitigates individual model biases, these foundational pointwise approaches fail to capture the fundamental reality of a motor race: drivers do not finish in absolute isolation; they finish in a strict relative order.

### **The Limitations of Pointwise Models in Motorsport**

Pointwise approaches look at a single instance at a time, training a standard regressor or classifier to predict a target variable, and subsequently sorting the outputs to form a ranking.11 In a pointwise regression model trained on Formula 1 data, the algorithm attempts to minimize the Root Mean Square Error (RMSE) or Mean Absolute Error (MAE) between the predicted finishing position and the actual finishing position.12 If the model predicts Driver A to finish 9.2 and Driver B to finish 10.4, it ranks Driver A ahead of Driver B.

However, pointwise models are blind to the competitive dynamics between instances.11 They do not penalize the inversion of two highly competitive drivers differently than the inversion of a race leader and a backmarker. Furthermore, classification models predicting a 20-class output face severe data sparsity issues; the probability of a midfield driver finishing exactly 10th is often statistically indistinguishable from their probability of finishing 9th or 11th, leading to flattened probability distributions that confuse EV calculations.5

### **Transitioning to Learning to Rank (LTR) Methodologies**

Learning to Rank (LTR) is a class of supervised machine learning algorithms specifically designed to sort a list of items based on relevance, making it inherently superior for race outcome prediction.15 In the context of motorsport, the "query" is the specific Grand Prix weekend, and the "documents" are the 20 participating drivers. The algorithm's objective is to order the drivers in a way that minimizes inversions relative to the actual finishing order, directly capturing the zero-sum nature of the race where one driver's advancement necessitates another's demotion.11

LTR approaches are categorized into three distinct methodologies, with pairwise and listwise approaches offering significant advantages for predicting specific finishing positions 11:

| LTR Methodology | Operational Mechanism | Application to F1 Race Prediction |
| :---- | :---- | :---- |
| **Pointwise** | Evaluates a single driver at a time to predict an absolute score or position. | Current repository state. Fails to account for inter-driver interactions and mutual exclusivity of finishing positions.11 |
| **Pairwise** | Evaluates pairs of drivers (e.g., Driver A vs. Driver B). The loss function penalizes the model if a driver who finished lower is ranked higher than a driver who finished ahead of them.11 | Highly effective for midfield predictions. Focuses the model on resolving close battles (e.g., the battle for P10) by minimizing local inversions.11 |
| **Listwise** | Evaluates the entire list of 20 drivers simultaneously, optimizing directly for ranking metrics like Normalized Discounted Cumulative Gain (NDCG) over the whole grid.13 | Optimal for overall race prediction. Treats the race as a single cohesive event, though mathematically more complex to implement.19 |

For predicting the 10th place finisher, a pairwise or listwise LTR approach is mathematically optimal.17 The LambdaMART algorithm—an adaptation of the LambdaRank framework applied to Gradient Boosted Decision Trees (GBDT)—stands out as the industry standard for these types of ranking problems.21

LambdaMART bypasses the difficulty of defining a smooth, differentiable loss function for discrete ranking metrics like NDCG. Instead, it defines the gradients (the "Lambdas") directly.23 During training, LambdaMART calculates a proxy gradient for each pair of drivers in a race.21 Crucially, it scales this gradient by the change in the target metric (such as NDCG) that would result from swapping the two drivers.22 This means the model learns to prioritize getting the ranking right in regions where errors are most heavily penalized.

### **Implementation via XGBRanker and LightGBM**

Both XGBoost and LightGBM offer native, highly optimized support for LambdaMART and pairwise ranking through their respective Python libraries.13

Integrating these models into the existing Python pipeline requires a structural modification to how the data is fed into the algorithms. Unlike standard regressors, LTR models require a qid (query ID) or group array that explicitly tells the algorithm which rows belong to which race.21 Without this grouping, the model would attempt to rank a driver's performance in the 2024 Monaco Grand Prix against a driver's performance in the 2025 Italian Grand Prix, completely destroying the contextual validity of the data.25

When utilizing XGBRanker, the objective function rank:ndcg should be selected.21 This objective utilizes the surrogate gradient derived from the NDCG metric, prioritizing the correct ordering of the entire list.21 Alternatively, rank:pairwise can be utilized to strictly minimize pairwise loss.21 In LightGBM, the lambdarank objective achieves a similar listwise evaluation that is reduced to a pointwise score for aggregation.28 By transitioning the Random Forest and baseline XGBoost regressors to their LTR equivalents, the ensemble will naturally correct for the mutual exclusivity of race positions, providing a much higher-fidelity estimate of the drivers competing precisely around the P10 boundary.5

## **Probability Calibration for Expected Value Maximization**

The repository's transition to an Expected Value (EV) architecture is a highly strategic approach for isolating the 10th place finisher.4 By generating a probability distribution across all 20 finishing positions for each driver, the system can apply a specific scoring vector (such as fantasy points or a custom mathematical weight centered on P10) and select the driver with the highest EV.4 However, this advanced methodology introduces a critical mathematical dependency: the absolute accuracy, or calibration, of the predicted probabilities.31

### **The Dangers of Uncalibrated Tree-Based Models**

Tree-based ensemble models, which form the core of the current pipeline (Random Forest classifiers, XGBoost classifiers), are notoriously poorly calibrated out-of-the-box.33 While they are exceptional at discriminating between classes (e.g., determining that Driver A is more likely to finish 10th than Driver B), the actual probability values they output are often distorted.31

Random Forests, for instance, calculate probabilities by averaging the class predictions across all trees. Because it is exceedingly rare for 100% of trees to vote for a single minority class, Random Forests systematically underestimate the probability of highly likely events and overestimate the probability of highly unlikely events.31 They push probability estimates away from 0 and 1, resulting in an artificially flattened distribution.31

In the context of the F1-p10-2026 model, if an uncalibrated Random Forest predicts a 15% chance of a driver finishing 10th, but the true empirical frequency of that specific prediction scenario is only 5%, the EV calculation becomes fundamentally distorted.32 In sports betting and fantasy sports optimization, relying on uncalibrated models leads to negative expected value in the long run, as the algorithm will confidently select suboptimal drivers based on inflated probability tails.31

### **Advanced Calibration Techniques for Multi-Class Problems**

To rectify this vulnerability, external probability calibration must be applied to the multi-class outputs before the EV calculations are performed.33 "A probabilistic classifier is well-calibrated if among test instances receiving a predicted probability vector p, the class distribution is (approximately) p".31

For binary classification, Platt Scaling (fitting a logistic regression model to the classifier outputs) or Isotonic Regression (fitting a non-parametric monotonic function) are standard procedures.31 However, the F1-p10 problem is a 20-class problem (positions 1 through 20). Calibrating multi-class probabilities is mathematically complex because the calibrated probabilities must remain a valid simplex—they must all be positive and sum to exactly 1 without introducing secondary biases.36

Several approaches should be evaluated and integrated into the pipeline:

1. **One-vs-Rest (OvR) Platt Scaling:** The most accessible method utilizing the Python scikit-learn library. The CalibratedClassifierCV wrapper automatically breaks the 20-class problem into 20 distinct binary calibration tasks (e.g., P10 vs. Not-P10) and then couples the calibrated forecasts, normalizing them to ensure they sum to 1\.36 This provides immediate improvements over raw tree probabilities.  
2. **Multi-Class Log-Loss Optimization (MCLLO):** A more advanced, state-of-the-art methodology that calibrates the entire vector of predicted probabilities simultaneously.38 MCLLO maximizes a likelihood function across all classes, ensuring that the relationships between adjacent ordinal positions (e.g., P9, P10, P11) are preserved during calibration.34  
3. **Dirichlet Calibration:** Directly calibrates probabilities within the simplex space using Dirichlet distributions, avoiding the artifacts sometimes introduced by reducing multi-class problems to binary tasks.36

Implementing a calibration layer (such as wrapping the rf\_clf model in CalibratedClassifierCV(method='isotonic', cv=5)) will drastically reduce variance in predictions.37 By ensuring that the probability mass assigned to the P10 outcome reflects strict empirical reality, the Expected Value optimizer will confidently select the driver with the highest true mathematical probability of achieving the desired result.31

## **Pre-Race Feature Engineering: Mining the Midfield**

Algorithm selection dictates how the model learns, but feature engineering dictates what the model can learn. Predicting the 10th place finisher relies on identifying minute performance differentials in the most congested segment of the grid. The current model utilizes an impressive 44 features, including track-specific VSC rates, season completeness, and historical qualifying density.4 However, relying purely on historical trends and basic qualifying results fails to capture the dynamic telemetry that dictates a specific Sunday's outcome.

### **Disentangling Qualifying Position from Starting Grid Penalties**

A foundational requirement of the user prompt is that predictions are made *after qualifying but before the race start*. The strongest historical predictor of race finishing position is the qualifying position, boasting a correlation coefficient (C) of 0.741 and a Spearman's Rho of 0.763 for the 2010-2023 era.3

However, a critical distinction must be made between the *qualifying position* (the pure measure of a car's single-lap pace) and the *starting grid position* (where the car actually lines up on Sunday).40 In modern Formula 1, drivers frequently incur grid penalties for changing power unit components (ICE, Turbocharger, MGU-K) beyond their seasonal allocation, receiving 5-place or 10-place drops.41 Drivers may also receive penalties for impeding others during qualifying or starting from the pit lane to break parc fermé rules and change suspension setups.42

If a model uses starting grid position as a proxy for raw pace, it will fundamentally misunderstand the capability of a penalized driver. A driver who qualifies 3rd but starts 13th due to an engine penalty possesses a car capable of podium pace; they will effortlessly cut through the midfield and are highly unlikely to finish 10th.44 Conversely, the drivers who inherit elevated grid positions (moving from 11th to 10th) are statistically likely to regress to their natural pace over a race distance.

The legacy Ergast API, previously a staple for F1 data, is deprecated and struggles with the nuance between qualifying and grid data.45 The predictive pipeline must rely on modern alternatives like the Jolpica API or OpenF1.47 Feature engineering must explicitly include both qualifying\_position (as a proxy for absolute car capability) and grid\_position (as a proxy for track position and traffic probability). A derived feature, grid\_penalty\_delta (Grid Position minus Qualifying Position), serves as a potent indicator of drivers who are out of position and likely to disrupt the midfield order during the race.49

### **Quantifying Race Pace via Free Practice 2 (FP2) Long Runs**

While qualifying measures single-lap pace on minimal fuel, the race is dictated by high-fuel race pace and tire degradation.50 The correlation between qualifying position and race outcome drops precipitously outside the top 5; 92% of top 5 finishers start in the top 10, but the midfield is highly volatile.52 A driver who qualifies 8th but suffers from severe thermal tire degradation will plummet through the field, often falling past the P10 boundary.54

Free Practice 2 (FP2) is traditionally utilized by teams to conduct heavy-fuel "long runs" to simulate race stints.55 Integrating FP2 long-run data via the FastF1 Python library provides a direct, pre-race proxy for Sunday's performance.56 FastF1 provides sub-millisecond telemetry, allowing models to extract lap-by-lap data for every driver.57

Rather than utilizing simple averages of FP2 times, advanced modeling requires extracting specific pace parameters. By extracting the lap times from FP2 stints (filtering out in-laps, out-laps, and traffic-affected anomalies), Bayesian state-space models or robust linear regressions can be fitted to the time series.59 This extracts two distinct, highly predictive variables:

1. **Base Pace (Intercept):** The theoretical pace of the car on fresh tires with a full race fuel load.60  
2. **Degradation Rate (Slope):** The time lost per lap due to thermal and physical tire wear.59

By calculating the delta between a driver's FP2 Base Pace and the median Base Pace of the entire midfield (drivers qualifying P6-P15), the model gains a predictive feature that completely bypasses qualifying anomalies.55 If a driver qualifies 13th but exhibits a top-8 Base Pace and a low Degradation Rate in FP2, the model will correctly identify them as a prime candidate to move forward into the points.55

### **Pit Stop Execution and the Undercut Probability**

In the tightly congested midfield, where cars are often trapped in aerodynamic "dirty air" unable to pass on track, the "undercut" is the primary strategic weapon.54 An undercut involves pitting before a rival, bolting on fresh tires, and using the immediate grip advantage to set a faster "out-lap," thereby jumping the rival when they pit on the subsequent lap.54

The success of an undercut is highly sensitive to pit stop stationary time.64 A flawless 2.2-second stop versus a sluggish 3.5-second stop can dictate whether a driver emerges in clean air or trapped behind a slower backmarker.54 While the repository currently evaluates generic circuit-level pit stop counts, it lacks team-specific execution variance.4

An analysis of 2025 pit stop data reveals severe disparities in execution among midfield competitors. For stops under 6 seconds, Ferrari averaged 2.51s, Racing Bulls averaged 2.56s, McLaren averaged 2.88s, and Alpine averaged a dismal 3.11s.66

To capture this, an Expected\_Pit\_Time (xPT) metric must be engineered.67 This feature calculates the rolling median pit stop time for each constructor over the previous 10 races, heavily penalized by a Pit\_Variance metric (the standard deviation of the team's stops).67 Teams with a low xPT and tight variance possess a significantly higher probability of executing successful undercuts.62 If a driver qualifies P11 but drives for a team with top-tier pit execution (e.g., Racing Bulls), their probability of breaching the P10 zone increases geometrically compared to a driver at a team prone to mechanical fumbles.66

| 2025 Pit Stop Stationary Time Variance by Selected Teams (Sub-6 Second Stops) |
| :---- |
| **Constructor** |
| Ferrari |
| Racing Bulls |
| Red Bull Racing |
| McLaren |
| Alpine |

### **Blue Flag Interference and Lapping Proximity**

A deeply under-researched variable in midfield race prediction is the catastrophic time loss induced by blue flags. Current sporting regulations dictate that a driver about to be lapped must allow the faster car to pass at the earliest safe opportunity.69 Navigating a blue flag requires the slower driver to lift off the racing line, shedding vital tire temperature and losing between 1.5 to 3.0 seconds per event.70

In races where the leaders establish a dominant, unassailable pace, they will frequently lap the field up to P7 or P8.70 This creates severe asymmetrical penalties in the midfield. For example, if the race leader laps the 10th place car but fails to catch the 9th place car before the checkered flag, the driver in P10 loses 3 seconds of race time while the driver in P9 loses nothing. This time loss can instantly drop the P10 driver into the clutches of P11, entirely destroying their race strategy.70

A predictive feature, Blue\_Flag\_Vulnerability, can be modeled deterministically prior to the race. By comparing the pole sitter's Q3 qualifying time to the 10th place qualifier's Q2 time, an expected race pace delta per lap is established.61 Multiplying this pace delta by the total number of race laps yields the expected time gap at the end of the race. If this expected gap exceeds the time required for one full lap, the P10 driver is mathematically guaranteed to encounter blue flags.70 Models can use this boolean or continuous probability metric to penalize drivers starting in the lower half of the top 10 on short circuits (e.g., Monaco, Red Bull Ring) where lapping is highly probable.70

### **Reliability Modeling and the DNF Proxy**

A driver cannot finish 10th if they do not finish at all. Did Not Finish (DNF) outcomes fundamentally alter the midfield math; every car that retires ahead of P10 automatically promotes the rest of the field by one position.3 While the current repository explored DNF signal features (e.g., driver\_circuit\_dnf\_rate), they were temporarily excluded due to a lack of discriminative power during the highly reliable 2022-2025 era.4

However, predicting mechanical DNFs requires moving beyond static historical rates and incorporating Bayesian tracking of Power Unit (PU) component age.75 A Power Unit accumulates wear over its lifecycle. By querying the FIA documentation (or utilizing the Jolpica API's penalty endpoints) 47, the model can track how many races a specific internal combustion engine or turbocharger has completed. A driver running a PU on its 6th consecutive race weekend has an exponentially higher mechanical DNF probability than a driver running a fresh engine, particularly at circuits with high ambient temperatures or extended full-throttle zones (e.g., Monza, Bahrain).75 Integrating an Engine\_Mileage\_Risk feature interacts deeply with track-specific thermal load requirements to accurately forecast attrition.79

## **Adapting to the 2026 Formula 1 Regulatory Paradigm**

The 2026 Formula 1 regulations represent the most severe combined reset of chassis aerodynamics and power unit physics in the sport's modern history.7 Predictive models heavily reliant on historical data from the ground-effect era (2022-2025) will suffer massive degradation if not explicitly adjusted for these new technical directives.9 The integration of 2026-specific data requires an algorithmic understanding of three core transformations: Powertrain Equilibrium, Active Aerodynamics, and the Manual Override Mode.

### **Powertrain Equilibrium and Energy Management**

The 2026 power units shift from an internal combustion-dominant output to a 50/50 equilibrium between the Internal Combustion Engine (ICE) and the electrical Energy Recovery System (ERS).81 The ICE output drops significantly from \~600kW to \~400kW, while the MGU-K (Kinetic Motor Generator Unit) electrical output nearly triples from 120kW to 350kW.79

This shift transforms Formula 1 from a fuel-limited formula into an electrical energy management formula.83 Drivers will be forced to harvest up to 8.5MJ of energy per lap to feed the MGU-K.85 To achieve this, drivers must engage in "lift and coast" tactics into braking zones and utilize "super clipping"—a phenomenon where the car harvests electrical energy while still at full throttle on long straights, artificially limiting top speed to recharge the battery.83

For predictive modeling, this necessitates entirely new features. Historical track characteristics (e.g., straight-line length) will no longer correlate linearly with top speed. Instead, tracks must be categorized by an Energy\_Demand\_Rating.76 Circuits with exceptionally long straights but very few heavy braking zones (e.g., Jeddah, Las Vegas, Monza) will severely penalize cars with inefficient energy recovery systems, as they will simply run out of battery deployment halfway down the straight.76 A new feature tracking a team's theoretical MGU-K duty cycle efficiency per circuit will be an essential predictor for midfield dominance.80

### **Active Aerodynamics: X-Mode and Z-Mode**

The traditional Drag Reduction System (DRS), which artificially manufactured overtaking since 2011, is abolished in 2026\.74 It is replaced by a dynamic, full-time Active Aero system affecting both the front and rear wings.74 The system oscillates between two states controlled by the driver 74:

* **Z-Mode (Corner Mode):** The standard high-downforce configuration for cornering, where all wing elements are closed to provide maximum aerodynamic grip.74  
* **X-Mode (Straight Mode):** The low-drag configuration, where both front and rear wing flaps open to shed up to 40% of total vehicle drag, massively increasing straight-line acceleration.81

Crucially, because X-Mode is available to *all* cars on straights (not just those within one second of a leading car like the old DRS rules), the traditional "DRS train" that heavily dictated midfield finishing positions will be fundamentally altered.92 Cars will accelerate much faster, hitting their terminal aerodynamic velocity sooner, which places an even greater emphasis on the electrical deployment to make a pass.88 The model's current features related to overtaking difficulty and qualifying density must be recalibrated, as the slipstream effect changes entirely when the leading car is also in a low-drag configuration.94

### **Manual Override Mode (MOM) / Overtake Mode**

To replace the localized overtaking advantage previously provided by DRS, the 2026 regulations introduce the Manual Override Mode (MOM), officially designated in the sporting regulations as "Overtake Mode".86

If a chasing car is within one second of the leading car at a designated detection point (typically before a straight), the driver gains access to an additional 0.5MJ of electrical deployment for the subsequent lap.86 This allows the chasing driver to sustain the peak 350kW electrical output deeper into the straight before the system begins to taper off (de-rate), providing the necessary speed delta to execute a pass.88

This mechanic fundamentally alters the mathematics of the undercut and track-position defense. A driver emerging from the pits in P11, behind a slower car in P10, will utilize the Overtake Mode to force a pass.99 Models must incorporate an Override\_Effectiveness feature, calculated as a ratio of a circuit's high-speed sector lengths to the total lap length.94 Tracks where MOM is highly effective will see higher variance in finishing positions, severely reducing the predictive power of the starting grid position and rewarding drivers with superior race pace.94

### **Reliability and "Infant Mortality" Rates**

With entirely new power units, complex active aerodynamic actuators, structural chassis overhauls, and smaller batteries operating under extreme thermal loads (the MGU-K output generating massive heat rejection challenges), the 2026 season will inevitably experience a severe spike in mechanical DNF rates.79

In reliability engineering, new complex systems experience "infant mortality"—a high failure rate early in their lifecycle before manufacturing defects are ironed out.101 While the F1-p10-2026 repository currently down-weights DNF proxy features due to the high reliability of the 2024-2025 era 4, predicting mechanical DNFs is paramount for 2026\.101 New engine manufacturers entering the sport (Audi, Ford-Red Bull Powertrains) lack the decade of hybrid-era operational data possessed by Mercedes and Ferrari, making their components statistically more vulnerable to failure.8 The predictive model must assign a high "infant mortality" penalty to these new manufacturers during the early stages of the 2026 season.

| Technical Parameter | Pre-2026 Paradigm | 2026 Paradigm | Impact on Predictive Modeling Architecture |
| :---- | :---- | :---- | :---- |
| **Power Unit Split** | 80% ICE / 20% Electric | 50% ICE / 50% Electric | Race pace becomes dictated by Battery State of Charge and recovery; necessitates Energy\_Demand\_Rating track features.74 |
| **Aerodynamics** | Static with specific DRS Zones | Active (X-Mode/Z-Mode) | "DRS Train" logic collapses; overall drag reduced by 40%, altering slipstream physics and overtaking metrics.81 |
| **Overtaking Aid** | DRS (Drag Reduction via rear wing) | MOM (Energy Override via MGU-K) | Overtaking relies entirely on stored electrical energy (+0.5MJ); track-specific Override\_Effectiveness features required.88 |
| **Dimensions / Weight** | 800kg, 2000mm width, long wheelbase | 768kg, 1900mm width, shorter wheelbase | Nimbler cars may drastically increase overtaking probability in narrow street circuits (Monaco, Singapore).74 |
| **Reliability Profile** | Exceptionally High (mature tech) | High Initial Variance (infant mortality) | Early-season DNF probability spikes; requires Bayesian tracking of PU component age and manufacturer experience penalties.8 |

## **Ranked Recommendations for Model Optimization**

Based on the exhaustive analysis of the F1-p10-2026 repository architecture 4 and the impending 2026 regulatory shift, the following enhancements are recommended. They are rank-ordered by their expected impact on the predictive accuracy of the 10th-place finisher versus the implementation complexity required in a Python environment.

### **1\. Implement Probability Calibration on Multi-Class EV Outputs**

**Impact:** Very High | **Effort:** Low The immediate, mathematically critical constraint on the current Expected Value architecture is the uncalibrated nature of the tree-based probability distributions (Random Forest, XGBoost).33

* **Action:** Wrap the existing classification models in src/models.py using scikit-learn's CalibratedClassifierCV. Utilize the 'isotonic' method for larger datasets or 'sigmoid' (Platt Scaling) for smaller folds.35  
* **Rationale:** Calibrated probabilities will dramatically normalize the EV calculations. It ensures the model does not erroneously select a driver simply because an uncalibrated Random Forest over-confidently assigned a 15% probability to the P10 class when the empirical reality is 5%.32 This requires minimal pipeline reconstruction while yielding immediate gains in Log-Loss and Brier score metrics.33

### **2\. Transition Core Architecture to a Listwise Learning-to-Rank (LTR) Paradigm**

**Impact:** Very High | **Effort:** High Predicting an ordinal finishing position via independent pointwise regression fundamentally misaligns with the zero-sum, mutually exclusive reality of a motor race.5

* **Action:** Introduce XGBRanker or LGBMRanker into the model ensemble, utilizing the rank:ndcg (LambdaMART) objective function.13 Group the training data by Grand Prix by passing the race\_id as the qid parameter during the .fit() call.21  
* **Rationale:** LTR models learn the *relative* performance differences between drivers within the exact same race conditions. By optimizing for NDCG, the model intrinsically understands that promoting a driver to P10 necessitates demoting the current occupant to P11, capturing the localized competitive dynamics that standard regressors miss entirely.11

### **3\. Integrate FP2 Race Pace State-Space Modeling via FastF1**

**Impact:** High | **Effort:** High Qualifying data alone explains less than 20% of the variance for finishing positions outside the top 5, making it an unreliable sole predictor for P10.52

* **Action:** Utilize the fastf1 Python library to extract FP2 session lap times.56 Develop a preprocessing module to filter out anomalies (in/out laps, traffic, VSC periods) and fit a linear regression or Bayesian state-space model to extract a "Base Pace" intercept and a "Degradation Rate" slope for each driver.59  
* **Rationale:** This creates a direct, pre-race proxy for Sunday's performance. A driver who suffers a poor qualifying session (e.g., P14) but exhibits a top-8 Base Pace and low degradation in FP2 is the optimal statistical candidate for a P10 prediction.55

### **4\. Engineer 2026-Specific Energy and Override Proxies**

**Impact:** High | **Effort:** Medium The 2026 regulations render historical speed-trap data and drag coefficients obsolete.80

* **Action:** Before the 2026 season begins, map all 24 circuits based on "Energy Recovery Potential" (number of heavy braking zones) and "Override Effectiveness" (percentage of lap spent on straights where MOM can be deployed).76  
* **Rationale:** Teams with superior powertrain efficiency will excel on high-demand tracks.76 Incorporating these circuit-level modifiers prevents the model from erroneously applying 2024-2025 ground-effect logic to 2026 energy-management races. Furthermore, applying an "infant mortality" DNF penalty to new engine manufacturers (Audi, RBPT) will correct for early-season attrition.101

### **5\. Incorporate Pit Stop Execution (xPT) and Blue Flag Risk**

**Impact:** Medium | **Effort:** Medium Midfield battles are won in the pit lane and lost to lapping traffic.54

* **Action:** Create a rolling Constructor\_xPT feature tracking the median pit stop times and standard deviation for each team over the previous 10 races (filtering out \>6s anomalies).66 Concurrently, compute a Blue\_Flag\_Vulnerability boolean based on the qualifying pace delta between P1 and P10 multiplied by the number of race laps.61  
* **Rationale:** These features act as critical micro-modifiers. When the LTR model detects a dead heat between two drivers for the P10 prediction, superior pit stop variance (e.g., Ferrari vs. Alpine) or mathematical freedom from blue flag traffic will accurately serve as the tiebreaker, correctly simulating the mechanics of an undercut.54

## **Strategic Development Plan (Python / Claude Code Integration)**

To systematically integrate these recommendations into the Python environment without disrupting the existing production pipeline (v4.03) 4, the following phased development roadmap is proposed for implementation via Claude Code.

### **Phase 1: Mathematical Stabilization (Weeks 1-2)**

**Objective:** Calibrate existing classification models and refine the EV selection logic to establish a mathematically sound baseline.

1. **Script Updates:** Using Claude Code, modify src/models.py. Wrap the existing rf\_clf and xgb\_clf model instantiations with sklearn.calibration.CalibratedClassifierCV(estimator=model, method='isotonic', cv=5).37  
2. **Scoring Adjustment:** Review the SCORING\_VECTOR in config.py. Ensure the vector scales appropriately with the newly calibrated probability distributions, preventing artificial skewing toward extreme outcomes.  
3. **Validation:** Run scripts/06\_seasonal\_performance\_analysis.py. Use Claude Code to output a comparative analysis of the logarithmic loss (Log-Loss) and Brier scores between the uncalibrated v4.03 models and the newly calibrated models.33 Accept the changes only if Log-Loss decreases.

### **Phase 2: Implementation of Learning to Rank (Weeks 3-5)**

**Objective:** Introduce LambdaMART architecture to capture relative grid ordering and pairwise interactions.21

1. **Data Structuring:** Instruct Claude Code to update src/feature\_engineering.py. The script must ensure the dataset is strictly sorted by race\_id. Extract the race\_id array to be passed directly as the group or qid parameter to the LTR models.21  
2. **Model Integration:** Add LGBMRanker and XGBRanker classes to src/models.py. Set the objective for XGBoost to rank:ndcg and LightGBM to lambdarank.21  
3. **Ensemble Weights:** Recalibrate the season-stage adaptive weights (EARLY, MID, LATE) in config.py.4 The LTR models, by virtue of understanding relative position, should theoretically absorb the weight currently assigned to the less effective standard rf\_reg and ridge pointwise models.

### **Phase 3: Telemetry Integration via FastF1 (Weeks 6-8)**

**Objective:** Move beyond tabular API data by ingesting raw, pre-race telemetry to calculate true race pace.56

1. **Data Fetching:** Create a new script, scripts/01b\_fetch\_telemetry.py. Utilize the fastf1 library (fastf1.get\_session(year, race, 'FP2')) to download the session data.58  
2. **Pace Extraction:** Write a utility function in src/feature\_engineering.py that utilizes Pandas to filter pick\_quicklaps().56 Fit a scipy.stats linear regression to the tire age vs. lap time to calculate the fp2\_base\_pace (intercept) and fp2\_deg\_rate (slope).59 Handle missing data (e.g., wet sessions) by utilizing the 5-level fallback logic (population median) established in v3.72.4  
3. **Retraining:** Execute scripts/02\_build\_dataset.py and scripts/03\_train\_models.py to rebuild the parquet datasets and retrain the ensemble with the newly engineered pace and pit stop (xPT) features.4

### **Phase 4: 2026 Regulation Hardening (Pre-Season 2026\)**

**Objective:** Prepare the predictive model for the new physics, energy management, and engine formulas of the 2026 era.80

1. **Era Weights Revision:** Using Claude Code, update the Era-Stratified Sample Weights in config.py. The 2022-2025 ground-effect data must be down-weighted significantly (e.g., from 1.00 to 0.40) to prevent the model from overfitting to obsolete aerodynamic dependencies.4  
2. **Circuit Mapping:** Generate a static CSV matrix mapping the 24 circuits on the 2026 calendar for Override\_Effectiveness and Energy\_Demand\_Rating.76 Inject these as standard features in build\_dataset.py.  
3. **Testing Analysis:** Utilize the is\_pre\_season\_fast cold-start mitigation logic (from Bahrain 2026 testing).4 Apply heavy Bayesian priors for mechanical DNF probabilities specifically targeting the new power unit manufacturers (Audi, Ford-RBPT) to account for anticipated infant mortality rates.101

## **Conclusion**

Predicting the 10th place finisher in Formula 1 requires navigating the most volatile, hyper-competitive segment of the racing grid. The current F1-p10-2026 repository has successfully established a robust data infrastructure and an innovative Expected Value architecture. However, to achieve the next echelon of predictive accuracy, the fundamental algorithmic philosophy must shift from absolute position regression to relative listwise ranking via LambdaMART architectures.

By implementing Learning-to-Rank algorithms, enforcing strict multi-class probability calibration to validate EV calculations, and engineering highly contextual pre-race features—such as FP2 race pace state-space models, pit stop variance metrics, blue flag vulnerability, and 2026-specific active aero and energy management proxies—the predictive model will transcend basic historical pattern matching. It will possess a mechanistic understanding of how a race physically unfolds, allowing for the precise, mathematically rigorous extraction of value at the highly contested perimeter of the points-scoring positions.

#### **Works cited**

1. The beginner's guide to the F1 Drivers' Championship, accessed March 13, 2026, [https://www.formula1.com/en/latest/article/the-beginners-guide-to-the-f1-drivers-championship.53MjXJzTDxQnfxfoCLnxNZ](https://www.formula1.com/en/latest/article/the-beginners-guide-to-the-f1-drivers-championship.53MjXJzTDxQnfxfoCLnxNZ)  
2. Trapped in the middle: The psychology of the midfield \- DIVEBOMB Motorsport, accessed March 13, 2026, [https://www.dive-bomb.com/article/trapped-in-the-middle-the-psychology-of-the-midfield](https://www.dive-bomb.com/article/trapped-in-the-middle-the-psychology-of-the-midfield)  
3. Evaluating the Predictive Power of Qualifying Performance in Formula One Grand Prix, accessed March 13, 2026, [https://arxiv.org/html/2507.10966v1](https://arxiv.org/html/2507.10966v1)  
4. V380\_PLAN.md  
5. \[P\] How to predict F1 race results? : r/MachineLearning \- Reddit, accessed March 13, 2026, [https://www.reddit.com/r/MachineLearning/comments/1k3m3uc/p\_how\_to\_predict\_f1\_race\_results/](https://www.reddit.com/r/MachineLearning/comments/1k3m3uc/p_how_to_predict_f1_race_results/)  
6. My CONTROVERSIAL PREDICTIONS for the 2026 F1 Season \- Kym Illman, accessed March 13, 2026, [https://www.kymillman.com/blog/my-controversial-predictions-for-the-2026-f1-season/](https://www.kymillman.com/blog/my-controversial-predictions-for-the-2026-f1-season/)  
7. Everything you need to know about the new F1 rules for 2026, accessed March 13, 2026, [https://www.formula1.com/en/latest/article/everything-you-need-to-know-about-the-new-f1-rules-for-2026.48bv0VTxhIlhrQXmxercXk](https://www.formula1.com/en/latest/article/everything-you-need-to-know-about-the-new-f1-rules-for-2026.48bv0VTxhIlhrQXmxercXk)  
8. 2026 REGULATIONS EXPLAINED: All you need to know about F1's new power units, accessed March 13, 2026, [https://www.formula1.com/en/latest/article/2026-regulations-explained-all-you-need-to-know-about-f1s-new-power-units.14jfv7a36905uDJDdNyfQd](https://www.formula1.com/en/latest/article/2026-regulations-explained-all-you-need-to-know-about-f1s-new-power-units.14jfv7a36905uDJDdNyfQd)  
9. tomasz-solis/formula1: Prediction model \+ API for regulations between 2022 and 2025 (included). Using XGBoost model to predict quali and race performance. Not to be used for 2026 season (new technical regulations). \- GitHub, accessed March 13, 2026, [https://github.com/tomasz-solis/formula1](https://github.com/tomasz-solis/formula1)  
10. Data-driven pit stop decision support for Formula 1 using deep learning models \- Frontiers, accessed March 13, 2026, [https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1673148/full](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1673148/full)  
11. Pointwise vs. Pairwise vs. Listwise Learning to Rank | by Nikhil Dandekar \- Medium, accessed March 13, 2026, [https://medium.com/@nikhilbd/pointwise-vs-pairwise-vs-listwise-learning-to-rank-80a8fe8fadfd](https://medium.com/@nikhilbd/pointwise-vs-pairwise-vs-listwise-learning-to-rank-80a8fe8fadfd)  
12. Building an F1 prediction engine – Predictive Modelling Part I \- F1 predictor, accessed March 13, 2026, [https://www.f1-predictor.com/building-an-f1-prediction-engine-predictive-modelling-part-i/](https://www.f1-predictor.com/building-an-f1-prediction-engine-predictive-modelling-part-i/)  
13. Learning to Rank using XGBoost. sci-kit learn and Pandas | by Simon Lind | Predictly on Tech | Medium, accessed March 13, 2026, [https://medium.com/predictly-on-tech/learning-to-rank-using-xgboost-83de0166229d](https://medium.com/predictly-on-tech/learning-to-rank-using-xgboost-83de0166229d)  
14. Predicting Average Finishing Position of Drivers in Formula One \- Medium, accessed March 13, 2026, [https://medium.com/@turkiznazaba/predicting-average-finishing-position-of-drivers-in-formula-one-92358554501f](https://medium.com/@turkiznazaba/predicting-average-finishing-position-of-drivers-in-formula-one-92358554501f)  
15. Learning to rank \- Wikipedia, accessed March 13, 2026, [https://en.wikipedia.org/wiki/Learning\_to\_rank](https://en.wikipedia.org/wiki/Learning_to_rank)  
16. Introduction to Ranking Algorithms | Towards Data Science, accessed March 13, 2026, [https://towardsdatascience.com/introduction-to-ranking-algorithms-4e4639d65b8/](https://towardsdatascience.com/introduction-to-ranking-algorithms-4e4639d65b8/)  
17. Learning to Rank \- Data Science \- Numerai Forum, accessed March 13, 2026, [https://forum.numer.ai/t/learning-to-rank/454](https://forum.numer.ai/t/learning-to-rank/454)  
18. Who's Better? Who's Best? Pairwise Deep Ranking for Skill Determination \- Dima Damen, accessed March 13, 2026, [https://dimadamen.github.io/Skill/whos\_better\_whos\_best.pdf](https://dimadamen.github.io/Skill/whos_better_whos_best.pdf)  
19. Listwise Approach to Learning to Rank \- Theory and Algorithm, accessed March 13, 2026, [https://icml.cc/Conferences/2008/papers/167.pdf](https://icml.cc/Conferences/2008/papers/167.pdf)  
20. Listwise Learning to Rank by Exploring Unique Ratings | Request PDF \- ResearchGate, accessed March 13, 2026, [https://www.researchgate.net/publication/338758135\_Listwise\_Learning\_to\_Rank\_by\_Exploring\_Unique\_Ratings](https://www.researchgate.net/publication/338758135_Listwise_Learning_to_Rank_by_Exploring_Unique_Ratings)  
21. Learning to Rank — xgboost 3.3.0-dev documentation, accessed March 13, 2026, [https://xgboost.readthedocs.io/en/latest/tutorials/learning\_to\_rank.html](https://xgboost.readthedocs.io/en/latest/tutorials/learning_to_rank.html)  
22. LambdaMART Explained: The Workhorse of Learning-to-Rank \- Shaped.ai, accessed March 13, 2026, [https://www.shaped.ai/blog/lambdamart-explained-the-workhorse-of-learning-to-rank](https://www.shaped.ai/blog/lambdamart-explained-the-workhorse-of-learning-to-rank)  
23. LambdaMART in Depth \- Doug Turnbull, accessed March 13, 2026, [https://softwaredoug.com/blog/2022/01/17/lambdamart-in-depth](https://softwaredoug.com/blog/2022/01/17/lambdamart-in-depth)  
24. How to Build Multi-Objective Ranking Models with LightGBM \- Floating Bytes, accessed March 13, 2026, [https://saraswatmks.github.io/2026/02/lightgbm\_pairwise\_lambdarank.html](https://saraswatmks.github.io/2026/02/lightgbm_pairwise_lambdarank.html)  
25. Understanding Groups in scikit-learn for XGBoost Ranking \- Stack Overflow, accessed March 13, 2026, [https://stackoverflow.com/questions/68563115/understanding-groups-in-scikit-learn-for-xgboost-ranking](https://stackoverflow.com/questions/68563115/understanding-groups-in-scikit-learn-for-xgboost-ranking)  
26. The inner workings of the lambdarank objective in LightGBM \- Frank Fineis, accessed March 13, 2026, [https://ffineis.github.io/blog/2021/05/01/lambdarank-lightgbm.html](https://ffineis.github.io/blog/2021/05/01/lambdarank-lightgbm.html)  
27. xgboost ranking objectives pairwise vs (ndcg & map) \- Stack Overflow, accessed March 13, 2026, [https://stackoverflow.com/questions/63400523/xgboost-ranking-objectives-pairwise-vs-ndcg-map](https://stackoverflow.com/questions/63400523/xgboost-ranking-objectives-pairwise-vs-ndcg-map)  
28. Prediction for the Learn-to-Rank Application · Issue \#3326 · microsoft/LightGBM \- GitHub, accessed March 13, 2026, [https://github.com/microsoft/LightGBM/issues/3326](https://github.com/microsoft/LightGBM/issues/3326)  
29. Optimization for Maximizing the Expected Value of Order Statistics \- Finance Department, accessed March 13, 2026, [https://finance.business.uconn.edu/wp-content/uploads/sites/723/2019/04/Optimization-for-Maximizing-the-Expected-Value-of-Order-Statistics.pdf](https://finance.business.uconn.edu/wp-content/uploads/sites/723/2019/04/Optimization-for-Maximizing-the-Expected-Value-of-Order-Statistics.pdf)  
30. How I've been playing Fantasy F1 this season without any F1 knowledge: My Expected Value & Optimisation Model : r/fantasyF1 \- Reddit, accessed March 13, 2026, [https://www.reddit.com/r/fantasyF1/comments/o2vcej/how\_ive\_been\_playing\_fantasy\_f1\_this\_season/](https://www.reddit.com/r/fantasyF1/comments/o2vcej/how_ive_been_playing_fantasy_f1_this_season/)  
31. An Introduction to Calibration \- by Conor Walsh \- Medium, accessed March 13, 2026, [https://medium.com/@conorwalsh206/an-introduction-to-calibration-e5a9f56173af](https://medium.com/@conorwalsh206/an-introduction-to-calibration-e5a9f56173af)  
32. Machine learning for sports betting: should predictive models be optimised for accuracy or calibration? \- SciSpace, accessed March 13, 2026, [https://scispace.com/pdf/machine-learning-for-sports-betting-should-predictive-models-3oo0vvnc.pdf](https://scispace.com/pdf/machine-learning-for-sports-betting-should-predictive-models-3oo0vvnc.pdf)  
33. Calibrating Multi-Class Models \- Proceedings of Machine Learning Research, accessed March 13, 2026, [https://proceedings.mlr.press/v152/johansson21a/johansson21a.pdf](https://proceedings.mlr.press/v152/johansson21a/johansson21a.pdf)  
34. Designing Sports Betting Systems in R: Bayesian Probabilities, Expected Value, and Kelly Logic | R-bloggers, accessed March 13, 2026, [https://www.r-bloggers.com/2026/02/designing-sports-betting-systems-in-r-bayesian-probabilities-expected-value-and-kelly-logic/](https://www.r-bloggers.com/2026/02/designing-sports-betting-systems-in-r-bayesian-probabilities-expected-value-and-kelly-logic/)  
35. Probability Calibration for 3-class Classification in Scikit Learn \- GeeksforGeeks, accessed March 13, 2026, [https://www.geeksforgeeks.org/machine-learning/probability-calibration-for-3-class-classification-in-scikit-learn/](https://www.geeksforgeeks.org/machine-learning/probability-calibration-for-3-class-classification-in-scikit-learn/)  
36. How to calibrate with multiclass classification problem? \- Stats StackExchange, accessed March 13, 2026, [https://stats.stackexchange.com/questions/543897/how-to-calibrate-with-multiclass-classification-problem](https://stats.stackexchange.com/questions/543897/how-to-calibrate-with-multiclass-classification-problem)  
37. Probability Calibration for 3-class classification — scikit-learn 1.8.0 documentation, accessed March 13, 2026, [https://scikit-learn.org/stable/auto\_examples/calibration/plot\_calibration\_multiclass.html](https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_multiclass.html)  
38. Multiclass Calibration Assessment and Recalibration of Probability Predictions via the Linear Log Odds Calibration Function \- arXiv, accessed March 13, 2026, [https://arxiv.org/html/2602.18573v1](https://arxiv.org/html/2602.18573v1)  
39. Machine learning for sports betting: should forecasting models be optimised for accuracy or calibration? \- ResearchGate, accessed March 13, 2026, [https://www.researchgate.net/publication/369184023\_Machine\_learning\_for\_sports\_betting\_should\_forecasting\_models\_be\_optimised\_for\_accuracy\_or\_calibration](https://www.researchgate.net/publication/369184023_Machine_learning_for_sports_betting_should_forecasting_models_be_optimised_for_accuracy_or_calibration)  
40. Formula One API \- Postman, accessed March 13, 2026, [https://documenter.getpostman.com/view/11586746/SztEa7bL](https://documenter.getpostman.com/view/11586746/SztEa7bL)  
41. The beginner's guide to F1 penalties, accessed March 13, 2026, [https://www.formula1.com/en/latest/article/the-beginners-guide-to-f1-penalties.5lne3FfE8IXpGOagsmfq90](https://www.formula1.com/en/latest/article/the-beginners-guide-to-f1-penalties.5lne3FfE8IXpGOagsmfq90)  
42. EXPLAINED: Pit lane starts \- Why drivers start from the back and how they fight forwards \- F1, accessed March 13, 2026, [https://www.formula1.com/en/latest/article/explained-pit-lane-starts-why-drivers-start-from-the-back.6GUzecVqR4rbSwNT79OGG3](https://www.formula1.com/en/latest/article/explained-pit-lane-starts-why-drivers-start-from-the-back.6GUzecVqR4rbSwNT79OGG3)  
43. Bianca Bustamante receives post-qualifying penalty \- F1 Academy, accessed March 13, 2026, [https://www.f1academy.com/Latest/6WV3VuWrVi5mbpPHKdXjPt/bianca-bustamante-receives-post-qualifying-penalty](https://www.f1academy.com/Latest/6WV3VuWrVi5mbpPHKdXjPt/bianca-bustamante-receives-post-qualifying-penalty)  
44. FIA post-qualifying press conference \- Russia \- F1, accessed March 13, 2026, [https://www.formula1.com/en/latest/article/fia-post-qualifying-press-conference-russia.e6izhVGHVkq4uNKd5RoV5](https://www.formula1.com/en/latest/article/fia-post-qualifying-press-conference-russia.e6izhVGHVkq4uNKd5RoV5)  
45. What information is only available from Ergast · theOehrly Fast-F1 · Discussion \#618 \- GitHub, accessed March 13, 2026, [https://github.com/theOehrly/Fast-F1/discussions/618](https://github.com/theOehrly/Fast-F1/discussions/618)  
46. Ergast F1 Database Error? : r/F1Technical \- Reddit, accessed March 13, 2026, [https://www.reddit.com/r/F1Technical/comments/hu6mac/ergast\_f1\_database\_error/](https://www.reddit.com/r/F1Technical/comments/hu6mac/ergast_f1_database_error/)  
47. jolpica-f1/README.md at main \- GitHub, accessed March 13, 2026, [https://github.com/jolpica/jolpica-f1/blob/main/README.md](https://github.com/jolpica/jolpica-f1/blob/main/README.md)  
48. OpenF1 API \- Real-time and historical Formula 1 data \- GitHub, accessed March 13, 2026, [https://github.com/br-g/openf1](https://github.com/br-g/openf1)  
49. Formula 1 Analysis in R with f1dataR: Lap Times, Pit Stops, and Driver Performance, accessed March 13, 2026, [https://www.r-bloggers.com/2026/03/formula-1-analysis-in-r-with-f1datar-lap-times-pit-stops-and-driver-performance/](https://www.r-bloggers.com/2026/03/formula-1-analysis-in-r-with-f1datar-lap-times-pit-stops-and-driver-performance/)  
50. As a fan, what is more impressive to you, qualifying pace or race pace? : r/F1Discussions, accessed March 13, 2026, [https://www.reddit.com/r/F1Discussions/comments/1pmrpgi/as\_a\_fan\_what\_is\_more\_impressive\_to\_you/](https://www.reddit.com/r/F1Discussions/comments/1pmrpgi/as_a_fan_what_is_more_impressive_to_you/)  
51. Race Vs. Quali Fast Laps: Where the Difference Lies \- YouTube, accessed March 13, 2026, [https://www.youtube.com/watch?v=\_bkz1K\_RwDs](https://www.youtube.com/watch?v=_bkz1K_RwDs)  
52. F1 Qualifying Times And Final Position What's the Relationship? \- RPubs, accessed March 13, 2026, [https://rpubs.com/Dannyc16/1173689](https://rpubs.com/Dannyc16/1173689)  
53. \[OC\] Formula 1: Probability of Grand Prix result by qualifying position \- Reddit, accessed March 13, 2026, [https://www.reddit.com/r/dataisbeautiful/comments/rgt8wz/oc\_formula\_1\_probability\_of\_grand\_prix\_result\_by/](https://www.reddit.com/r/dataisbeautiful/comments/rgt8wz/oc_formula_1_probability_of_grand_prix_result_by/)  
54. Undercut vs Overcut: The Real Strategy War in Modern F1 \- grandprix247, accessed March 13, 2026, [https://www.grandprix247.com/formula-1-news/undercut-vs-overcut-the-real-strategy-war-in-formula-1](https://www.grandprix247.com/formula-1-news/undercut-vs-overcut-the-real-strategy-war-in-formula-1)  
55. Predicting F1 Podiums with 78% Accuracy Using Machine Learning and Real Race Data, accessed March 13, 2026, [https://ipolishchuk22.medium.com/predicting-f1-podiums-with-78-accuracy-using-machine-learning-and-real-race-data-0de3bcd6d2c4](https://ipolishchuk22.medium.com/predicting-f1-podiums-with-78-accuracy-using-machine-learning-and-real-race-data-0de3bcd6d2c4)  
56. Team Pace Comparison \- FastF1 3.6.1, accessed March 13, 2026, [https://docs.fastf1.dev/gen\_modules/examples\_gallery/plot\_team\_pace\_ranking.html](https://docs.fastf1.dev/gen_modules/examples_gallery/plot_team_pace_ranking.html)  
57. FastF1 Playbook: 10 Notebooks to Master Formula 1 Data in 2026 | by Raul Garcia, accessed March 13, 2026, [https://medium.com/formula-one-forever/fastf1-playbook-10-notebooks-to-master-formula-1-data-in-2026-23c347a462b3](https://medium.com/formula-one-forever/fastf1-playbook-10-notebooks-to-master-formula-1-data-in-2026-23c347a462b3)  
58. Timing and Telemetry Data \- fastf1.core, accessed March 13, 2026, [http://docs.fastf1.dev/core.html](http://docs.fastf1.dev/core.html)  
59. (PDF) A State-Space Approach to Modeling Tire Degradation in Formula 1 Racing, accessed March 13, 2026, [https://www.researchgate.net/publication/398227195\_A\_State-Space\_Approach\_to\_Modeling\_Tire\_Degradation\_in\_Formula\_1\_Racing](https://www.researchgate.net/publication/398227195_A_State-Space_Approach_to_Modeling_Tire_Degradation_in_Formula_1_Racing)  
60. A State-Space Approach to Modeling Tire Degradation in Formula 1 Racing \- arXiv.org, accessed March 13, 2026, [https://arxiv.org/pdf/2512.00640](https://arxiv.org/pdf/2512.00640)  
61. The F1 Pace Delta \- Machine Learning \- Profound Academy, accessed March 13, 2026, [https://profound.academy/machine-learning/the-f1-pace-delta-2vHA5gsrlAwwffvGNFvh](https://profound.academy/machine-learning/the-f1-pace-delta-2vHA5gsrlAwwffvGNFvh)  
62. Data-driven pit stop decision support for Formula 1 using deep learning models \- PMC, accessed March 13, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12626961/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12626961/)  
63. Virtual Strategy Engineer: Using Artificial Neural Networks for Making Race Strategy Decisions in Circuit Motorsport \- MDPI, accessed March 13, 2026, [https://www.mdpi.com/2076-3417/10/21/7805](https://www.mdpi.com/2076-3417/10/21/7805)  
64. A Thorough Analysis of the Pit Stop Strategy in Formula 1 \- Statathlon, accessed March 13, 2026, [https://statathlon.com/analysis-of-the-pit-stop-strategy-in-f1/](https://statathlon.com/analysis-of-the-pit-stop-strategy-in-f1/)  
65. F1 Pit Stop Strategies Explained | PDF \- Scribd, accessed March 13, 2026, [https://www.scribd.com/presentation/958071054/F1-Pit-Stop-Analysis-Story-With-Visuals](https://www.scribd.com/presentation/958071054/F1-Pit-Stop-Analysis-Story-With-Visuals)  
66. F1 2025 Pitstop Data : r/formula1 \- Reddit, accessed March 13, 2026, [https://www.reddit.com/r/formula1/comments/1q984of/f1\_2025\_pitstop\_data/](https://www.reddit.com/r/formula1/comments/1q984of/f1_2025_pitstop_data/)  
67. 2025 F1 Season: Pit Stop Power Rankings (Rounds 1 \- 12\) : r/F1Technical \- Reddit, accessed March 13, 2026, [https://www.reddit.com/r/F1Technical/comments/1m734x3/2025\_f1\_season\_pit\_stop\_power\_rankings\_rounds\_1\_12/](https://www.reddit.com/r/F1Technical/comments/1m734x3/2025_f1_season_pit_stop_power_rankings_rounds_1_12/)  
68. 2025 F1 Season: Pit Stop Power Rankings (rounds 1 \- 18), accessed March 13, 2026, [https://f1pace.com/p/2025-f1-season-pit-stop-power-rankings-rounds-1-18/](https://f1pace.com/p/2025-f1-season-pit-stop-power-rankings-rounds-1-18/)  
69. The beginner's guide to F1 flags, accessed March 13, 2026, [https://www.formula1.com/en/latest/article/the-beginners-guide-to-formula-1-flags.T5DqOqbWI6S4Va8Y5yMld](https://www.formula1.com/en/latest/article/the-beginners-guide-to-formula-1-flags.T5DqOqbWI6S4Va8Y5yMld)  
70. \[Discussion\] Do blue flags affect the racing for midfield and backmarkers too much? \- Reddit, accessed March 13, 2026, [https://www.reddit.com/r/formula1/comments/l3l5ep/discussion\_do\_blue\_flags\_affect\_the\_racing\_for/](https://www.reddit.com/r/formula1/comments/l3l5ep/discussion_do_blue_flags_affect_the_racing_for/)  
71. How do teams analyse F1 race strategy? | The MIA, accessed March 13, 2026, [https://www.schoolofraceengineering.co.uk/blog/post/15986/how-do-teams-analyse-f1-race-strategy/](https://www.schoolofraceengineering.co.uk/blog/post/15986/how-do-teams-analyse-f1-race-strategy/)  
72. Qualifying Deltas for Drivers Post 2000 : r/F1Discussions \- Reddit, accessed March 13, 2026, [https://www.reddit.com/r/F1Discussions/comments/1o1xwsh/qualifying\_deltas\_for\_drivers\_post\_2000/](https://www.reddit.com/r/F1Discussions/comments/1o1xwsh/qualifying_deltas_for_drivers_post_2000/)  
73. October | 2014 \- f1metrics, accessed March 13, 2026, [https://f1metrics.wordpress.com/2014/10/](https://f1metrics.wordpress.com/2014/10/)  
74. The Hidden Strategy Shift in F1 2026: What the New Regulations Really Mean for Fans, accessed March 13, 2026, [https://www.autoracing1.com/pl/474209/the-hidden-strategy-shift-in-f1-2026-what-the-new-regulations-really-mean-for-fans/](https://www.autoracing1.com/pl/474209/the-hidden-strategy-shift-in-f1-2026-what-the-new-regulations-really-mean-for-fans/)  
75. 2020-01-0545 Strategy for Optimizing an F1 Car's Performance based on FIA Regulations, accessed March 13, 2026, [https://radar.brookes.ac.uk/radar/file/e8b7ba77-70d1-4f72-92a3-8caed44adbec/1/Optimizing%20F1%20car%20performance%20-%202020-01-0545%20-%20Bopaiah%20Samuel.pdf](https://radar.brookes.ac.uk/radar/file/e8b7ba77-70d1-4f72-92a3-8caed44adbec/1/Optimizing%20F1%20car%20performance%20-%202020-01-0545%20-%20Bopaiah%20Samuel.pdf)  
76. Predicting the impact of FIA 2026 Power Unit Regulations on the performance of Formula 1 Car \- Resource summary | openEQUELLA, accessed March 13, 2026, [https://radar.brookes.ac.uk/radar/items/9b174f33-59e1-449d-8673-00eebca8aeaf/1/](https://radar.brookes.ac.uk/radar/items/9b174f33-59e1-449d-8673-00eebca8aeaf/1/)  
77. harningle/fia-doc: Parse FIA PDF documents to get race data \- GitHub, accessed March 13, 2026, [https://github.com/harningle/fia-doc](https://github.com/harningle/fia-doc)  
78. marcll/f1-fia-doc-parser: Python-based tool for retrieving and processing FIA Formula One World Championship documents and its related information \- GitHub, accessed March 13, 2026, [https://github.com/marcll/f1-fia-doc-parser](https://github.com/marcll/f1-fia-doc-parser)  
79. Off the Grid: Why F1's Thermal Limits Demand Simulation Flexibility \- Modelon, accessed March 13, 2026, [https://modelon.com/blog/off-the-grid-why-f1s-thermal-limits-demand-simulation-flexibility/](https://modelon.com/blog/off-the-grid-why-f1s-thermal-limits-demand-simulation-flexibility/)  
80. F1 2026 Rules Explained: New Cars & Changes Explained \- Coffee Corner Motorsport, accessed March 13, 2026, [https://coffeecornermotorsport.com/f1-2026-rules/](https://coffeecornermotorsport.com/f1-2026-rules/)  
81. F1 2026 new rules: How does overtake mode work and what is active aero?, accessed March 13, 2026, [https://www.independent.co.uk/f1/f1-2026-new-rules-overtake-mode-active-aero-drs-b2934133.html](https://www.independent.co.uk/f1/f1-2026-new-rules-overtake-mode-active-aero-drs-b2934133.html)  
82. 7 things you need to know about the 2026 F1 engine regulations | Formula 1®, accessed March 13, 2026, [https://www.formula1.com/en/latest/article/more-efficient-less-fuel-and-carbon-net-zero-7-things-you-need-to-know-about.ZhtzvU3cPCv8QO7jtFxQR](https://www.formula1.com/en/latest/article/more-efficient-less-fuel-and-carbon-net-zero-7-things-you-need-to-know-about.ZhtzvU3cPCv8QO7jtFxQR)  
83. Five things Max Verstappen hates about the new F1 cars \- Drive.com.au, accessed March 13, 2026, [https://www.drive.com.au/caradvice/five-things-max-verstappen-hates-about-the-new-f1-cars/](https://www.drive.com.au/caradvice/five-things-max-verstappen-hates-about-the-new-f1-cars/)  
84. Formula 1 Is Now An Energy Game \- Above The Yellow Line \- ATYL Media, accessed March 13, 2026, [https://www.abovetheyellowline.com/episode/formula-1-is-now-an-energy-game/](https://www.abovetheyellowline.com/episode/formula-1-is-now-an-energy-game/)  
85. 2026 FORMULA 1 TECHNICAL REGULATIONS \- FIA, accessed March 13, 2026, [https://www.fia.com/sites/default/files/fia\_2026\_formula\_1\_technical\_regulations\_issue\_8\_-\_2024-06-24.pdf](https://www.fia.com/sites/default/files/fia_2026_formula_1_technical_regulations_issue_8_-_2024-06-24.pdf)  
86. EXPLAINED: The key terms for Formula 1's new-for-2026 rules, accessed March 13, 2026, [https://www.formula1.com/en/latest/article/explained-the-new-key-terms-for-formula-1s-new-for-2026-rules.3T5BU6TC9quGcIpGzoWkY0](https://www.formula1.com/en/latest/article/explained-the-new-key-terms-for-formula-1s-new-for-2026-rules.3T5BU6TC9quGcIpGzoWkY0)  
87. ️ F1 Race Prediction Simulator \- GitHub, accessed March 13, 2026, [https://github.com/mehmetkahya0/f1-race-prediction](https://github.com/mehmetkahya0/f1-race-prediction)  
88. I think the 2026 rules are going to hurt overtaking a lot : r/F1Technical \- Reddit, accessed March 13, 2026, [https://www.reddit.com/r/F1Technical/comments/1hazqls/i\_think\_the\_2026\_rules\_are\_going\_to\_hurt/](https://www.reddit.com/r/F1Technical/comments/1hazqls/i_think_the_2026_rules_are_going_to_hurt/)  
89. F1 2026 Regulations Explained: Every new rule, car change and key questions answered, accessed March 13, 2026, [https://www.gpfans.com/en/f1-news/1077929/f1-2026-regulations-explained/](https://www.gpfans.com/en/f1-news/1077929/f1-2026-regulations-explained/)  
90. How F1 2026's new active aero will work without DRS \- Motor Sport Magazine, accessed March 13, 2026, [https://www.motorsportmagazine.com/articles/single-seaters/f1/how-f1-2026s-new-active-aero-will-work-without-drs/](https://www.motorsportmagazine.com/articles/single-seaters/f1/how-f1-2026s-new-active-aero-will-work-without-drs/)  
91. Hybrids, Advanced Sustainable Fuels & Speed: Why F1 2026 Matters to Space Enthusiasts, accessed March 13, 2026, [https://universemagazine.com/en/hybrids-advanced-sustainable-fuels-speed-why-f1-2026-matters-to-space-enthusiasts/](https://universemagazine.com/en/hybrids-advanced-sustainable-fuels-speed-why-f1-2026-matters-to-space-enthusiasts/)  
92. The beginner’s guide to the 2026 Formula 1 regulations, accessed March 13, 2026, [https://www.formula1.com/en/latest/article/the-beginners-guide-to-the-2026-regulations.6j0tS0hrHG2T01tpmK6XYz](https://www.formula1.com/en/latest/article/the-beginners-guide-to-the-2026-regulations.6j0tS0hrHG2T01tpmK6XYz)  
93. Only 2 races left with DRS : r/formula1 \- Reddit, accessed March 13, 2026, [https://www.reddit.com/r/formula1/comments/1p7hk19/only\_2\_races\_left\_with\_drs/](https://www.reddit.com/r/formula1/comments/1p7hk19/only_2_races_left_with_drs/)  
94. F1 2026's straightline mode will slash drag \- with one major catch \- RaceTeq.com, accessed March 13, 2026, [https://www.raceteq.com/articles/2025/10/2026-straightline-mode-replacing-drs-simulations](https://www.raceteq.com/articles/2025/10/2026-straightline-mode-replacing-drs-simulations)  
95. GLOSSARY: The Term Changes You Need To Know For 2026 | Atlassian Williams F1 Team, accessed March 13, 2026, [https://www.williamsf1.com/articles/d24a2a68-9eac-494b-aff0-7c05b260d384/glossary-the-term-changes-you-need-to-know-for-2026](https://www.williamsf1.com/articles/d24a2a68-9eac-494b-aff0-7c05b260d384/glossary-the-term-changes-you-need-to-know-for-2026)  
96. \[F1\] It's all change in 2026\! Here's the key technical information as we head into a new generation... \- Reddit, accessed March 13, 2026, [https://www.reddit.com/r/formula1/comments/1powb62/f1\_its\_all\_change\_in\_2026\_heres\_the\_key\_technical/](https://www.reddit.com/r/formula1/comments/1powb62/f1_its_all_change_in_2026_heres_the_key_technical/)  
97. F1 2026 explained: Active Aero, Boost, Recharge and Overtake Mode \- RaceTeq.com, accessed March 13, 2026, [https://www.raceteq.com/articles/2026/01/f1-2026-explained-active-aero-boost-recharge-and-overtake-mode](https://www.raceteq.com/articles/2026/01/f1-2026-explained-active-aero-boost-recharge-and-overtake-mode)  
98. How overtaking will change in F1 under the new 2026 regulations \- Motorsport.com, accessed March 13, 2026, [https://www.motorsport.com/f1/news/fri-how-overtaking-will-change-in-f1-2026-under-new-regulations/10766622/](https://www.motorsport.com/f1/news/fri-how-overtaking-will-change-in-f1-2026-under-new-regulations/10766622/)  
99. Why Lando Norris predicts 2026 F1 rules “will create more chaos” in racing \- Motorsport.com, accessed March 13, 2026, [https://www.motorsport.com/f1/news/lando-norris-predicts-f1-2026-rules-will-create-more-chaos-in-racing/10795590/](https://www.motorsport.com/f1/news/lando-norris-predicts-f1-2026-rules-will-create-more-chaos-in-racing/10795590/)  
100. F1 is CHANGING EVERYTHING in 2026\! | Active Aero, Overtake mode & More \- Kym Illman, accessed March 13, 2026, [https://www.kymillman.com/blog/f1-is-changing-everything-in-2026-active-aero-overtake-mode-amp-more/](https://www.kymillman.com/blog/f1-is-changing-everything-in-2026-active-aero-overtake-mode-amp-more/)  
101. The F1 'differentiator' tipped to return in 2026 \- The Race, accessed March 13, 2026, [https://www.the-race.com/formula-1/f1-2026-new-rules-reliability/](https://www.the-race.com/formula-1/f1-2026-new-rules-reliability/)  
102. The car brands powering every 2026 F1 team, accessed March 13, 2026, [https://www.drive.com.au/news/what-engine-every-f1-team-is-using-in-2026/](https://www.drive.com.au/news/what-engine-every-f1-team-is-using-in-2026/)  
103. 2026 F1 — how would you rank all 11 teams based on rumors? \- Reddit, accessed March 13, 2026, [https://www.reddit.com/r/F1Discussions/comments/1ouych4/2026\_f1\_how\_would\_you\_rank\_all\_11\_teams\_based\_on/](https://www.reddit.com/r/F1Discussions/comments/1ouych4/2026_f1_how_would_you_rank_all_11_teams_based_on/)  
104. Will ICE components be more reliable in 2026? : r/F1Technical \- Reddit, accessed March 13, 2026, [https://www.reddit.com/r/F1Technical/comments/1psobbr/will\_ice\_components\_be\_more\_reliable\_in\_2026/](https://www.reddit.com/r/F1Technical/comments/1psobbr/will_ice_components_be_more_reliable_in_2026/)  
105. Can Lewis Hamilton strike in Shanghai? Russell and Leclerc set for tough challenge, accessed March 13, 2026, [https://www.planetf1.com/features/f1-2026-chinese-grand-prix-hamilton-prediction](https://www.planetf1.com/features/f1-2026-chinese-grand-prix-hamilton-prediction)  
106. Advanced Topics — LightGBM 4.4.0 documentation, accessed March 13, 2026, [https://lightgbm.readthedocs.io/en/v4.4.0/Advanced-Topics.html](https://lightgbm.readthedocs.io/en/v4.4.0/Advanced-Topics.html)  
107. From smaller cars to a bigger budget cap – 12 rule changes you need to know in 2026 \- F1, accessed March 13, 2026, [https://www.formula1.com/en/latest/article/from-smaller-cars-to-a-bigger-budget-cap-12-rule-changes-you-need-to-know-in.56uUTFhB0z5j3iZfhC0rGP](https://www.formula1.com/en/latest/article/from-smaller-cars-to-a-bigger-budget-cap-12-rule-changes-you-need-to-know-in.56uUTFhB0z5j3iZfhC0rGP)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEUAAAAXCAYAAABdy4LVAAACzklEQVR4Xu2XTchNQRjHH6GIfEeivCxIFr6KlCSxeBUJC2UtGyWEUlaysLAgSRJZSD43ShbSXSiKlMKChZeULF4LZaXw/7/PPM6cOTP3nrmHsji/+nXPzJnzzJmPMzNXpKWlpeX/YxLcDy/C43B2+XaUifAK3A3nij7jO6UoWop/zKV9WP4O3OyuQ1mX0SvWX2EefAn3wHFwEL6Fq/xCEfiyH+CvhGdducXwDTzontnr0gvdfbISfpdqDJPPkDqxGjMGXoK33bVxEj6A4728EDbkheiI+d6Dr0U7myPckXIs1sP6rrtrsgU+l2qsDnwIJ0v9WI1ZAD/Do0H+dtGRY8NTsCE2gsZYeAFucumN8Ce8+qeEwvqGRUeeHBAt68OOuAWXuHTdWI2xisJOYYM5bblepOBLzAny2EmH4SiXtjixhjCf98kKOK24PfL8KbjDy6sbqzFWUapTwvxuLId3RUfY6NUQrg0xtorOOM48o99Y2VjAsPG5ncKXvwF3Bfnz4Sd4TYrZY+tAKv50+AiuCfKzY/GlZkl1K4s5E47Wx0Z6NxYwt1PYgCGpftd8+SPwHRxweavhV0nHZ8e+gjOC/OxYy6S6cqc8I0XQVONT+TH4sufgMzg1uEc4APvgR9GOOw9PiMYP1wHuKtxdwt3QyInVN2vhD6k23jqFu1AvOKLcgjtSPmR1g2eYYanOLKaZH64b3UjF6hvuHkNSHLQM7iJhRfw8+b2HLIXfJD66TJ8WPa3aAmznjdjZYlB0MHhOCsmN1Tec+px+T6WoiOvTTSlXtAh+EV3ouOD5pM4PxF7af47bLM9G3K1COBipzzY3ViPYGfdFdw+O1GX4WPT/jMFr/hWw06WPjW6sU8gh+ET0zMMZ+R5uKJUoSC38Rk6sxnAB4+l1p/u13akOnFnrRXe2GJyNA3AbXCfls0fIBNHTcOpPXk6slpaWlpZ/xW+KC8QfAm/cwgAAAABJRU5ErkJggg==>