import sys
import time
import random
import pandas as pd
import numpy as np
import multiprocessing as mp



transactions_file = r"C:\Users\HP\Desktop\Genetic-Algorithm\data\Customer_Ledger_Entries_FULL_parsed.xlsx"
targets_file = r"C:\Users\HP\Desktop\Genetic-Algorithm\data\KH_Bank_parsed_parsed.xlsx"

# Load already parsed Excel files directly into DataFrames
transactions_df = pd.read_excel(transactions_file)
targets_df = pd.read_excel(targets_file)

# ✅ Extract relevant columns
transaction_amounts = transactions_df["Amount"].dropna().tolist()
target_amounts = targets_df["Statement.Entry.Amount.Value"].dropna().tolist()

# ✅ Keep Customer Names for filtering
transaction_names = transactions_df["Customer Name"].astype(str).tolist()
target_debtor_names = targets_df["Statement.Entry.EntryDetails.TransactionDetails.RelatedParties.Debtor.Name"].astype(str).tolist()

class SubsetSumGA:
    def __init__(self, numbers, targets, pop_size=100, gen_count=500, mut_rate=0.05, tourn_size=3, elite_size=5):
        self.numbers = np.array(numbers)
        self.targets = targets
        self.pop_size = pop_size
        self.gen_count = gen_count
        self.mut_rate = mut_rate
        self.tourn_size = tourn_size
        self.elite_size = elite_size

        self.total_sum = np.sum(self.numbers)
        self.avg_number = np.mean(self.numbers)

    def _fuzzy_ratio(self, achieved_sum, target):
        diff = abs(target - achieved_sum)
        return max(0, 100 * (1 - diff / max(abs(target), 1)))

    def _create_individual(self, target=None):
        chromosome = np.zeros(len(self.numbers), dtype=int)
        k = random.randint(1, min(10, len(self.numbers)))  # ✅ subset ≤ 10
        indices = np.random.choice(len(self.numbers), k, replace=False)
        chromosome[indices] = 1
        return chromosome

    def _fitness(self, chromosome, target):
        subset_sum = np.dot(chromosome, self.numbers)
        fuzzy = self._fuzzy_ratio(subset_sum, target)
        fitness = fuzzy

        if subset_sum == target:
            fitness += 1000
        subset_size = np.sum(chromosome)
        if subset_size == 0:
            fitness -= 100
        elif subset_size > 10:  # ✅ enforce subset ≤ 10
            fitness -= 50
        return fitness

    def _selection(self, population, fitnesses):
        idx = random.sample(range(len(population)), self.tourn_size)
        best_idx = max(idx, key=lambda i: fitnesses[i])
        return population[best_idx].copy()

    def _crossover(self, parent1, parent2):
        if random.random() < 0.7:
            point = random.randint(1, len(parent1) - 1)
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
        else:
            mask = np.random.randint(0, 2, len(parent1))
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)

        # ✅ enforce max 10 genes
        for child in (child1, child2):
            if np.sum(child) > 10:
                ones_idx = np.where(child == 1)[0]
                np.random.shuffle(ones_idx)
                child[ones_idx[10:]] = 0
        return child1.astype(int), child2.astype(int)

    def _mutate(self, chromosome, target=None):
        chromosome = chromosome.copy()
        for i in range(len(chromosome)):
            if random.random() < self.mut_rate:
                chromosome[i] = 1 - chromosome[i]
        if np.sum(chromosome) > 10:  # ✅ enforce ≤10
            ones_idx = np.where(chromosome == 1)[0]
            np.random.shuffle(ones_idx)
            chromosome[ones_idx[10:]] = 0
        return chromosome

    def run_ga_for_target(self, target, verbose=False):
        population = [self._create_individual(target) for _ in range(self.pop_size)]

        best_solution = (target, [], None, False)
        best_fitness = float("-inf")

        for generation in range(self.gen_count):
            fitnesses = [self._fitness(ind, target) for ind in population]
            max_idx = np.argmax(fitnesses)
            best_ind = population[max_idx]
            best_sum = np.dot(best_ind, self.numbers)

            if fitnesses[max_idx] > best_fitness:
                best_fitness = fitnesses[max_idx]
                best_solution = (target, self.numbers[best_ind == 1].tolist(),
                                 best_sum, best_sum == target)

            if abs(best_sum - target) <= 0.1 * abs(target):  # ✅ stop if within 10%
                break

            elite_indices = np.argsort(fitnesses)[-self.elite_size:]
            new_population = [population[i].copy() for i in elite_indices]

            while len(new_population) < self.pop_size:
                p1, p2 = self._selection(population, fitnesses), self._selection(population, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                new_population.extend([self._mutate(c1, target), self._mutate(c2, target)])
            population = new_population[:self.pop_size]

        return best_solution


# ✅ Direct Matching Function
def direct_match_with_name_filter(transactions_df, targets_df):
    matches = []
    for _, target_row in targets_df.iterrows():
        target_value = target_row["Statement.Entry.Amount.Value"]
        debtor_name = str(target_row["Statement.Entry.EntryDetails.TransactionDetails.RelatedParties.Debtor.Name"])

        # Filter transactions only for this debtor name
        filtered_txns = transactions_df.loc[
            transactions_df["Customer Name"].astype(str) == debtor_name, "Amount"
        ].dropna().tolist()

        for amount in filtered_txns:
            if amount == target_value:
                matches.append((target_value, amount, debtor_name))
    return matches


def run_ga_task(task):
        target, debtor_name = task
        filtered_numbers = transactions_df.loc[
            transactions_df["Customer Name"].astype(str) == debtor_name, "Amount"
        ].dropna().tolist()

        if not filtered_numbers:
            return None

        ga = SubsetSumGA(numbers=filtered_numbers, targets=[target], pop_size=50, gen_count=200)
        return ga.run_ga_for_target(target)
# ✅ MAIN EXECUTION (WRAPPED)
# ============================
if __name__ == "__main__":

    # === Direct Match with Name Filter ===
    start_time = time.time()
    direct_results = direct_match_with_name_filter(transactions_df, targets_df)
    direct_time = time.time() - start_time

    print("\n=== Direct Match Results (Name Filtered) ===")
    print(f"Matches found: {len(direct_results)}")
    for match in direct_results[:10]:
        print(f"Target: {match[0]}  |  Match: {match[1]} | Debtor: {match[2]}")
    print(f"Execution time: {direct_time:.4f} seconds")


    # === Genetic Algorithm with Name Filter ===
    start_time = time.time()
    ga_results = []

    print("\n=== Genetic Algorithm Progress (Name Filtered) ===")

    # Pair each target with its debtor name
    tasks = list(zip(target_amounts, target_debtor_names))

    # Worker function for multiprocessing
    

    with mp.Pool(mp.cpu_count()) as pool:
        for i, result in enumerate(pool.imap_unordered(run_ga_task, tasks), start=1):
            if result and result[2] is not None:
                ga_results.append(result)
                print(f"[{i}/{len(tasks)}] "
                      f"Target: {result[0]} | Achieved: {result[2]} | Subset Size: {len(result[1])}")

    ga_time = time.time() - start_time

    # === Final Results with Accuracy Metrics ===
    print("\n=== Genetic Algorithm Results Summary ===")
    print(f"Total targets: {len(ga_results)}")
    print(f"Exact matches found: {sum(1 for r in ga_results if r[3])}")

    # Show first 5 results
    for result in ga_results[:5]:
        print(f"Target: {result[0]}  |  Achieved: {result[2]}  |  Exact: {result[3]}  |  Subset Size: {len(result[1])}")

    # Calculate percentage error for each result
    errors = []
    within_tolerance = 0
    for target, subset, achieved, exact, *_ in ga_results:
        if achieved is not None and target != 0:
            error_pct = abs((achieved - target) / target) * 100
            errors.append(error_pct)
            if error_pct <= 10:  # within ±10% tolerance
                within_tolerance += 1

    avg_error = np.mean(errors) if errors else float('nan')
    accuracy = (within_tolerance / len(ga_results) * 100) if ga_results else 0

    print(f"\nExecution time: {ga_time:.4f} seconds")
    print(f"Average % error across targets: {avg_error:.2f}%")
    print(f"Accuracy (within ±10%): {accuracy:.2f}%")
