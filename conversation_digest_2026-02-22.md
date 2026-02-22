# Conversation Digest (2026-02-22)

## 1) Current codebase status (CRM)

- Repository core:
  - `main_CRM.py` as liquid CRM entry script.
  - `Modules/CRM_module.py` as the core CRMP class.
  - `CRM-gas.py` as gas-oriented extension script.
- Core model logic in `CRMP`:
  - Production prediction is recursive over time.
  - Terms include decay term + injection response term + optional pressure term.
  - Parameters fitted by `scipy.optimize.minimize(..., method='SLSQP')`.
- Important fix already applied:
  - `q0` optimization variable now correctly updates `self.qp0` in `CRMP`.
  - Fitted parameters are written back into model state after fit.

## 2) What the referenced paper contributes

- Paper focus: revised CaRM for large-diameter shallow-bore helical GHE.
- Main point:
  - Keep lumped RC-style thermal model efficiency.
  - Improve borehole-core heat transfer representation to better match CFD.
- Validation style:
  - Field data + CFD as reference.
  - Revised model reduced mismatch vs previous CaRM.

## 3) Why move from reservoir CRM to thermal CaRM

- Your current CRM is a dynamic coupling framework; this is reusable.
- Required migration is variable/physics substitution:
  - From flow-rate response to temperature/heat-flow response.
  - From injector-producer connectivity to thermal coupling/resistance network.
- Result:
  - Retain speed and interpretability.
  - Gain thermal physical consistency.

## 4) Gas direction summary

- Gas adaptation should prioritize pseudo-pressure-domain driving and pressure-aware constraints.
- Suggested practical path:
  - Build hybrid baseline (decline + CRM interference).
  - Add pressure/IRP consistency when Pwf data are available.
  - Keep hard constraints for identifiability and robustness.

## 5) Research framing summary (PhD-level)

- Stronger framing than “CRM + LSTM”:
  - Generalized RC network as unified dynamic framework.
  - Physics-constrained learning for residual/unmodeled dynamics.
  - Cross-domain transfer (geothermal + hydrogen storage/charging network).
- Target contribution stack:
  - Unified model.
  - Identifiable parameter estimation.
  - Forecast-control closed loop with measurable engineering value.

## 6) Agent integration summary

- Best role split:
  - Agent handles pipeline orchestration and governance.
  - Physics model remains numerical core.
- Minimal useful agent stack:
  - Data Agent, Model Agent, Physics Agent, Forecast Agent.
- Avoid:
  - LLM replacing the physics core directly.

## 7) GCAM coupling summary

- Feasible through soft coupling (scenario-parameter exchange), not full hard coupling initially.
- Direction:
  - GCAM provides macro demand/price/policy scenarios.
  - Engineering model provides feasible supply-cost-performance envelopes back to GCAM.

## 8) Immediate implementation roadmap

- Step 1: thermalize CRM (target = heat output prediction).
- Step 2: add pressure/thermal constraints and uncertainty intervals.
- Step 3: connect to optimization/control and optional macro coupling.
- Step 4: package as repeatable experiments for publication quality.

