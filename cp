import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
from docplex.cp.model import *
from docplex.cp.config import context
context.solver.local.execfile = "/Applications/CPLEX_Studio2211/cpoptimizer/bin/arm64_osx/cpoptimizer"
from data import data_read
import numpy as np
import csv
from itertools import chain

np.random.seed(0)

instances, s, tc, machines = data_read()


def CP_opt(job_list, setup_time, machines, tc_list, dc):
    job_number = len(job_list)
    job = {}  # dict for interval variables
    u = {}  # dict for integer variables
    tar = {}  # dict for tardiness variable
    model = CpoModel(name="splitting")

    # Maximum makespan variable for Agent B
    c_b = integer_var(name="c_b", min=0, max=10000)

    # Distribution center assignment variables
    z = [integer_var(min=1, max=dc, name=f"z_{i}") for i in range(1, job_number + 1)]

    # Create interval variables for each job on each machine
    for i in range(1, job_number + 1):
        if job_list[i - 1].agent == "A":
            tar[i] = integer_var(name=f'T_{i}', min=0, max=50000)

        for j in range(1, machines + 1):
            job[i, j] = interval_var(
                name=f'x_{i}_{j}',
                start=(0, 20000),
                end=(0, 20000),
                optional=True,
                size=(0, job_list[i - 1].total_time[j - 1])
            )

            u[i, j] = integer_var(
                name=f'u_{i}_{j}',
                min=0,
                max=job_list[i - 1].amount
            )

    # Create dummy jobs at the start of each machine sequence
    for j in range(1, machines + 1):
        job[0, j] = interval_var(name=f'x_0_{j}', start=0, end=(0, 20000), size=0)

    # Constraints for Agent A and B jobs
    for i in range(1, job_number + 1):
        if job_list[i - 1].agent != "C":
            for j in range(1, machines + 1):
                # Link job size to amount processed
                model.add(
                    job_list[i - 1].process_time[j - 1] * u[i, j] * presence_of(job[i, j]) ==
                    size_of(job[i, j])
                )

                # If job is present on machine, at least 1 unit must be processed
                model.add(u[i, j] >= presence_of(job[i, j]))

                # Amount processed cannot exceed total amount if job is present
                model.add(presence_of(job[i, j]) * job_list[i - 1].amount >= u[i, j])

    # Total amount constraint: sum of splits equals total amount
    for i in range(1, job_number + 1):
        if job_list[i - 1].agent != "C":
            model.add(
                sum(u[i, j] * presence_of(job[i, j]) for j in range(1, machines + 1)) ==
                job_list[i - 1].amount
            )

    # Sequence constraints with setup times
    sequence = {}
    for j in range(1, machines + 1):
        machine_jobs = [job[i, j] for i in range(job_number + 1)]
        seq = sequence_var(machine_jobs, name=f'seq_{j}')
        sequence[j] = seq

        # No overlap with setup times
        model.add(no_overlap(seq, setup_time[j - 1], is_direct=True))

        # Dummy job must be first
        model.add(first(seq, job[0, j]))

    # Tardiness constraints for Agent A
    for i in range(1, job_number + 1):
        if job_list[i - 1].agent == "A":
            for j in range(1, machines + 1):
                model.add(
                    tar[i] >=
                    end_of(job[i, j]) +
                    presence_of(job[i, j]) * element(tc_list, (z[i - 1] - 1) * machines + j - 1) -
                    job_list[i - 1].due_date
                )

    # Makespan constraints for Agent B
    for i in range(1, job_number + 1):
        if job_list[i - 1].agent == "B":
            for j in range(1, machines + 1):
                model.add(
                    c_b >=
                    end_of(job[i, j]) +
                    presence_of(job[i, j]) * element(tc_list, (z[i - 1] - 1) * machines + j - 1)
                )

    # Constraints for Agent C (pre-scheduled jobs)
    for i in range(1, job_number + 1):
        if job_list[i - 1].agent == "C":
            for j in range(1, machines + 1):
                if job_list[i - 1].assigned_machine == j:
                    model.add(presence_of(job[i, j]) == 1)
                    # Use <= instead of == to allow flexibility
                    model.add(end_of(job[i, j]) <= job_list[i - 1].completion_time)
                    model.add(presence_of(job[i, j]) * job_list[i - 1].process_time[j - 1] == size_of(job[i, j]))
                else:
                    model.add(presence_of(job[i, j]) == 0)

    # Search phases for better performance
    p1 = search_phase(
        vars=list(u.values()),
        varchooser=select_smallest(domain_size()),
        valuechooser=select_smallest(value_impact())
    )

    p2 = search_phase(vars=list(job.values()))

    # Objective: weighted sum of tardiness and makespan
    model.minimize(
        0.8 * sum(tar[i] for i in range(1, job_number + 1)
                  if job_list[i - 1].agent == "A") +
        0.2 * c_b
    )

    return model, tar, c_b


results = []

for idx, ins in enumerate(instances):
    print(f"Solving instance {idx + 1}/{len(instances)}...")

    # Count jobs
    number_job = 0
    order_no = 0
    for i in ins:
        if i.agent != "C":
            number_job += i.amount
            order_no += 1

    # Prepare data
    tc_list = list(chain.from_iterable(tc[idx]))
    dc = len(tc[idx])

    # Build model
    model, tar, c_b = CP_opt(ins, s[idx], machines[idx], tc_list, dc)

    start = time.time()

    # Solve with time limit
    sol = model.solve(TimeLimit=3600, LogVerbosity='Quiet')

    end = time.time()
    run_time = round(end - start)

    # Extract results
    if sol:
        obj = sol.get_objective_values()[0]

        # Calculate tardiness
        trd = 0
        for i in range(len(ins)):
            if ins[i].agent == "A":
                trd += sol.get_value(tar[i + 1])

        # Get makespan
        ms = sol.get_value(c_b)

        # Get gap
        gap = sol.get_objective_gap()
        solve_time = round(sol.get_solve_time())
        status = "Feasible"

        print(f"  Solution found: obj={obj:.2f}")
    else:
        obj = None
        trd = 0
        ms = 0
        gap = None
        solve_time = None
        status = "Infeasible"
        print(f"  No solution found")

    results.append({
        'Instance': idx + 1,
        'order': order_no,
        "job no": number_job,
        "Assigned job": len(ins) - order_no,
        "Machine_no": machines[idx],
        "DC": dc,
        "Obj": obj,
        "Tardiness": trd,
        "Makespan": ms,
        'Total Solve Time': run_time,
        'Solve Time': solve_time,
        "Gap": gap,
        "Status": status
    })

# Write results to file
file_path = 'results_cp.txt'
with open(file_path, 'w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')

    # Write header
    writer.writerow(["Instance", "Order", "Job No", "Assigned Job", "Machine No", "DC",
                     "Obj", "Tardiness", "Makespan", 'Total Solve Time', "Solve Time",
                     'Gap', 'Status'])

    # Write data rows
    for result in results:
        writer.writerow([
            result['Instance'], result['order'], result["job no"],
            result["Assigned job"], result["Machine_no"], result["DC"],
            result["Obj"], result["Tardiness"], result["Makespan"],
            result['Total Solve Time'], result['Solve Time'], result['Gap'],
            result['Status']
        ])

print(f"\nResults written to {file_path}")
print(f"Solved {len([r for r in results if r['Status'] == 'Feasible'])}/{len(results)} instances successfully")
