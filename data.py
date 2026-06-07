import numpy as np
import pandas as pd


class Job:
    """Represents a job in the scheduling system."""

    def __init__(self, job_id, amount, process_time, total_time, due_date,
                 agent, assigned_machine, completion_time):
        self.job_id = job_id
        self.amount = amount
        self.process_time = process_time  # Array of process times per machine
        self.total_time = total_time  # Total time (amount * process_time)
        self.due_date = due_date
        self.agent = agent
        self.assigned_machine = assigned_machine
        self.completion_time = completion_time

    def __repr__(self):
        return f'Job(id={self.job_id}, agent={self.agent}, machine={self.assigned_machine})'


def data_read(seed=None):
    """
    Generate scheduling instances with jobs and setup times.

    Returns:
        instances: List of job lists
        setup_instances: Setup time matrices for each instance
        tc: Transportation cost matrices
        machines: Number of machines per instance
    """
    if seed is not None:
        np.random.seed(seed)

    instances = []
    tc = []
    setup_instances = []
    machines = []

    order = 46  # Initial number of orders
    dc = [[1, 2, 3], [1, 2, 3]]  # Distribution centers configuration
    job_no = 10  # Initial job amount

    for j in range(2):
        machine_no = 2

        for z in range(3):
            for i in range(10):  # 10 instances per configuration
                # Generate scheduled jobs per machine
                scheduled_jobs = np.random.randint(
                    low=1,
                    high=int(order / machine_no) + 2,
                    size=machine_no
                ).tolist()

                job_list = []
                total_job_units = 0
                total_makespan = np.zeros(machine_no)  # Track makespan per machine

                # Generate Agent A and B jobs
                agent_b_id = 1
                for k in range(order):
                    total_job_units += job_no

                    # Generate process time for each machine
                    process_time = np.random.randint(low=2, high=5, size=machine_no)
                    total_time = job_no * process_time  # Total time per machine
                    total_makespan += total_time

                    # Assign agent
                    if k < order / 2:
                        agent = "A"
                        job_id = f"{k + 1}_{agent}"
                    else:
                        agent = "B"
                        job_id = f"{agent_b_id}_{agent}"
                        agent_b_id += 1

                    job = Job(
                        job_id=job_id,
                        amount=job_no,
                        process_time=process_time,
                        total_time=total_time,
                        due_date=0,
                        agent=agent,
                        assigned_machine=0,
                        completion_time=0
                    )
                    job_list.append(job)

                # Set due dates for Agent A jobs
                for job in job_list:
                    if job.agent == "A":
                        # Use max total_time across machines for due date calculation
                        max_time = np.max(job.total_time)
                        min_time = np.min(job.total_time)
                        # Make due dates more feasible
                        job.due_date = np.random.randint(
                            low=int(min_time + 20),
                            high=int((max_time + 40) * (total_job_units / machine_no))
                        )

                # Generate Agent C jobs (pre-scheduled) with more reasonable deadlines
                agent_c_id = 1
                for machine_idx, num_scheduled in enumerate(scheduled_jobs):
                    # Calculate average makespan for this machine
                    avg_makespan = total_makespan[machine_idx] / machine_no

                    # For Agent C jobs, use deadlines (not exact completion times)
                    # This gives the solver flexibility to find feasible sequences
                    for job_idx in range(num_scheduled):
                        job_id = f"{agent_c_id}_C"
                        agent_c_id += 1

                        # Generate process time
                        process_time = np.random.randint(low=1, high=4, size=machine_no)

                        # Set deadline based on position in sequence
                        # Earlier jobs get earlier deadlines, but with buffer
                        position_factor = (job_idx + 1) / num_scheduled

                        # Deadline: distribute across expected makespan with buffer
                        deadline = int(avg_makespan * position_factor * 1.2 +
                                       np.random.randint(10, 50))

                        job = Job(
                            job_id=job_id,
                            amount=1,
                            process_time=process_time,
                            total_time=process_time,  # For Agent C, total_time = process_time
                            due_date=0,
                            agent="C",
                            assigned_machine=machine_idx + 1,  # 1-indexed
                            completion_time=deadline  # Using as deadline, not exact time
                        )
                        job_list.append(job)

                instances.append(job_list)

                # Generate setup time matrices
                total_jobs = order + sum(scheduled_jobs)
                setup = []
                for machine_idx in range(machine_no):
                    setup_matrix = np.random.randint(
                        low=1,
                        high=4,
                        size=(total_jobs + 1, total_jobs + 1)
                    )
                    # Set diagonal to high value (no setup when job follows itself)
                    np.fill_diagonal(setup_matrix, 100)
                    # Set first column to high value (dummy job)
                    setup_matrix[:, 0] = 100
                    setup.append(setup_matrix.tolist())

                setup_instances.append(setup)

                # Generate transportation cost matrix
                tc_matrix = np.random.randint(
                    low=30,
                    high=60,
                    size=(dc[j][z], machine_no)
                )
                tc.append(tc_matrix)
                machines.append(machine_no)

        # Increase complexity for next iteration
        order += 4
        job_no += 4

    return instances, setup_instances, tc, machines


def export_to_excel(instances, tc, filename="output.xlsx"):
    """Export instances to Excel file."""
    with pd.ExcelWriter(filename) as writer:
        for ins_idx, ins in enumerate(instances):
            data = []

            for job in ins:
                if job.agent != "C":
                    row = {
                        "Job ID": job.job_id,
                        "Due Date": job.due_date,
                        "Assigned Machine": "None"
                    }
                else:
                    row = {
                        "Job ID": job.job_id,
                        "Deadline": job.completion_time,  # Now a deadline, not exact time
                        "Assigned Machine": job.assigned_machine
                    }

                # Add process times as separate columns
                for idx, time in enumerate(job.total_time):
                    row[f"pt {idx + 1}"] = time

                data.append(row)

            # Create DataFrame and export
            df_jobs = pd.DataFrame(data)
            df_jobs.to_excel(
                writer,
                sheet_name=f"Instance {ins_idx + 1}",
                index=False,
                startrow=0
            )

            # Add transportation cost matrix
            df_matrix = pd.DataFrame(tc[ins_idx])
            startcol = len(df_jobs.columns) + 2
            df_matrix.to_excel(
                writer,
                sheet_name=f"Instance {ins_idx + 1}",
                index=True,
                startrow=0,
                startcol=startcol
            )


