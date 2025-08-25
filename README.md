# Genetic Algorithm for Subset Sum in FinTech

## 📌 Project Overview
This project was developed as part of a FinTech assignment to address the **Subset Sum Problem** in the context of financial reconciliation.  

The task is to match **transaction amounts** from a company’s customer ledger with **target amounts** from bank statements.  
Two approaches were explored and compared:

1. **Brute Force Search**
   - Examines all possible subsets of transactions to find exact matches with targets.  
   - Computationally feasible only for very small datasets.  
   - With thousands of transactions, this results in **millions or billions of combinations**, making brute force impractical.

2. **Genetic Algorithm (GA)**
   - An optimization technique inspired by natural selection.  
   - Instead of checking all combinations, it **evolves subsets of transactions** over generations to approach target values.  
   - Scales efficiently and executes within **seconds**, even on large datasets.  

---

## 🚀 Why Genetic Algorithm?
The **brute force approach is not feasible** for financial datasets containing thousands of rows.  
For example:  
- 1,000 transactions → Over **2^1000 subsets** (astronomically large).  
- Impossible to compute within a reasonable time.  

The **Genetic Algorithm** solves this by:
- Searching intelligently instead of exhaustively.  
- Using probabilistic operators (selection, crossover, mutation) to quickly find subsets.  
- Producing **high accuracy** within practical runtime.  

---

## ⚙️ Features of the GA Implementation
Several enhancements were added to make the GA **faster** and **more efficient**:

- ✅ **Customer Name Filter** – restricts the search space by matching only transactions from the same customer/debtor name as in the target bank entry.  
- ✅ **Subset Size Limit (≤ 10)** – ensures subsets remain small, avoiding large and unnecessary combinations.  
- ✅ **Fuzzy Fitness Function** – rewards subsets that closely approximate the target, not just exact matches.  
- ✅ **Elitism** – best solutions are preserved across generations.  
- ✅ **Tournament Selection** – ensures stronger candidates are chosen for reproduction.  
- ✅ **Crossover & Mutation** – generate diversity while respecting the subset size limit.  
- ✅ **Multiprocessing** – runs targets in parallel across all CPU cores, significantly reducing runtime.  

---

## 📂 Project Structure
Genetic-Algorithm/
│
├── data/
│ ├── Customer_Ledger_Entries_FULL_parsed.xlsx
│ ├── KH_Bank_parsed_parsed.xlsx
│
├── genetic_algo.py # Main Python script (Genetic Algorithm + Direct Match)
├── README.md # Project documentation
├── requirements.txt # Python dependencies


---

## 📊 Comparison of Approaches
### Brute Force
- Checks every possible subset.  
- **Not feasible** beyond small datasets (millions of combinations).  
- Would take hours or days for real financial data.

### Genetic Algorithm
- Evolves subsets intelligently.  
- Runs in **seconds** on typical hardware (e.g., Intel i7).  
- Finds **exact matches** where possible and **close matches** (within ±10%) otherwise.  
- Accuracy improved by **customer name filtering** and **parallelization**.

---

## ▶️ How to Run
1. Clone or download this repository.  
2. Place your parsed Excel files inside the `data/` folder.  
3. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Example Output
### Direct Matches
=== Direct Match Results (Name Filtered) ===
Matches found: 10
Target: 1500000.0  |  Match: 1500000.0 | Debtor: ABC Ltd
Target: 5473217.0  |  Match: 5473217.0 | Debtor: XYZ Corp
...
Execution time: 0.3309 seconds
### Genetic Algorithm
=== Genetic Algorithm Progress (Name Filtered) ===
[1/38] Target: 2995679.0 | Achieved: 2995679.0 | Subset Size: 3
[2/38] Target: 3117847.0 | Achieved: 3117847.0 | Subset Size: 2
...

=== Genetic Algorithm Results Summary ===
Total targets: 38
Exact matches found: 10
Execution time: 5.40 seconds
Average % error across targets: 1.45%
Accuracy (within ±10%): 100

