# GUI and Tkinter-related imports
import tkinter as tk
from tkinter import messagebox, simpledialog, Toplevel, BOTH, font
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

# Data visualization and graph-related imports
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Image processing imports
from PIL import Image, ImageTk, ImageDraw

# System and utility imports
import os
import sys
import subprocess

# Scientific computing imports
import numpy as np
import heapq
import random

def potentiel_metra():

    # Create a new window for the MPM scheduler
    mpm_window = Toplevel(root)
    mpm_window.title("Metra Potential Method (MPM) Scheduler")
    mpm_window.state("zoomed")  # Set full screen mode

    # MPMApp logic
    class MPMApp:
        def __init__(self, root):
            self.root = root

            self.num_tasks_label = ttk.Label(root, text="Enter the number of tasks:")
            self.num_tasks_label.pack()

            self.num_tasks_entry = ttk.Entry(root)
            self.num_tasks_entry.pack()

            self.generate_button = ttk.Button(root, text="Generate Table", command=self.generate_table, bootstyle=SUCCESS)
            self.generate_button.pack()

            self.table_frame = ttk.Frame(root)
            self.table_frame.pack()

            self.critical_path_label = ttk.Label(root, text="", foreground="red")
            self.critical_path_label.pack()

        def generate_table(self):
            num_tasks = int(self.num_tasks_entry.get())
            self.tasks = []
            
            for i in range(num_tasks):
                task_id = f"T{i+1}"
                duration = random.randint(1, 10)
                if i == 0:
                    predecessors = []
                else:
                    predecessors = random.sample([f"T{j+1}" for j in range(i)], random.randint(0, i))
                self.tasks.append({"id": task_id, "duration": duration, "predecessors": predecessors})

            self.display_table()
            self.apply_mpm()

        def display_table(self):
            for widget in self.table_frame.winfo_children():
                widget.destroy()

            columns = ("ID", "Date au plus tot (ES)", "Date au plus tard (LS)", "Marge Total (Slack)", "Dependencies")
            self.treeview = ttk.Treeview(self.table_frame, columns=columns, show="headings")
            
            for col in columns:
                self.treeview.heading(col, text=col)
            
            for task in self.tasks:
                dependencies = ", ".join(task["predecessors"])
                self.treeview.insert("", "end", values=(task["id"], "", "", "", dependencies))

            self.treeview.pack()

        def apply_mpm(self):
            G = nx.DiGraph()
            
            for task in self.tasks:
                G.add_node(task["id"], duration=task["duration"])
                for pred in task["predecessors"]:
                    G.add_edge(pred, task["id"])

            es = {task["id"]: 0 for task in self.tasks}
            ef = {task["id"]: 0 for task in self.tasks}
            
            for task in nx.topological_sort(G):
                es[task] = max((ef[pred] for pred in G.predecessors(task)), default=0)
                ef[task] = es[task] + G.nodes[task]["duration"]

            lf = {task["id"]: ef[max(ef, key=ef.get)] for task in self.tasks}
            ls = {task["id"]: lf[task["id"]] - G.nodes[task["id"]]["duration"] for task in self.tasks}
            
            for task in reversed(list(nx.topological_sort(G))):
                lf[task] = min((ls[succ] for succ in G.successors(task)), default=lf[task])
                ls[task] = lf[task] - G.nodes[task]["duration"]

            slack = {task: ls[task] - es[task] for task in es}
            
            critical_path = [task for task in es if slack[task] == 0]

            for i, task in enumerate(self.tasks):
                es_val, ls_val, slack_val = es[task["id"]], ls[task["id"]], slack[task["id"]]
                dependencies = ", ".join(task["predecessors"])
                self.treeview.set(self.treeview.get_children()[i], column="Date au plus tot (ES)", value=es_val)
                self.treeview.set(self.treeview.get_children()[i], column="Date au plus tard (LS)", value=ls_val)
                self.treeview.set(self.treeview.get_children()[i], column="Marge Total (Slack)", value=slack_val)
                self.treeview.set(self.treeview.get_children()[i], column="Dependencies", value=dependencies)

            self.visualize_graph(G, critical_path)

        def visualize_graph(self, G, critical_path):
            # Clear the previous plot
            plt.clf()
            
            pos = nx.spring_layout(G)
            plt.figure(figsize=(12, 8))
            
            nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold")
            
            edge_labels = {(u, v): f"{G.nodes[v]['duration']}" for u, v in G.edges()}
            
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            
            critical_edges = [(u, v) for u, v in zip(critical_path[:-1], critical_path[1:])]
            
            nx.draw_networkx_edges(G, pos, edgelist=critical_edges, edge_color="r", width=2)

            plt.title("MPM Diagram with Critical Path Highlighted")
            
            # Embed the matplotlib figure in the tkinter window
            canvas = FigureCanvasTkAgg(plt.gcf(), master=self.root)
            
            # Clear previous canvas if exists
            if hasattr(self, 'canvas_widget'):
                self.canvas_widget.destroy()
            
            self.canvas_widget = canvas.get_tk_widget()
            self.canvas_widget.pack(fill=ttk.BOTH, expand=True)
            
            canvas.draw()

            critical_path_str = " -> ".join(critical_path)
            self.critical_path_label.config(text=f"Critical Path: {critical_path_str}")

    # Initialize the MPMApp in the new window
    app = MPMApp(mpm_window)

def ford_fulkerson():
    def open_graph_window():
        # Ask user for the number of nodes (sommets)
        num_nodes = simpledialog.askinteger("Input", "How many nodes do you want?", minvalue=1)
        if num_nodes is None:
            return

        # Generate a random directed graph with 50% probability of connection between nodes and random capacities
        graph = nx.DiGraph()
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and random.random() < 0.5:  # 50% probability of connection
                    capacity = random.randint(1, 20)  # Random capacity between 1 and 20
                    graph.add_edge(i, j, capacity=capacity)

        # Create a new window
        graph_window = Toplevel(root)
        graph_window.title("Random Graph")

        # Create a matplotlib figure for the initial graph
        fig, ax = plt.subplots(figsize=(6, 4))
        pos = nx.spring_layout(graph)
        capacities = nx.get_edge_attributes(graph, 'capacity')
        nx.draw(graph, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500, font_size=10, ax=ax)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=capacities)
        ax.set_title("Random Graph with Capacities")

        # Embed the matplotlib figure in the tkinter window
        canvas = FigureCanvasTkAgg(fig, master=graph_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)
        canvas.draw()

        # Add an Apply button
        apply_button = ttk.Button(graph_window, text="Apply Ford-Fulkerson", command=lambda: apply_ford_fulkerson(graph, graph_window, pos))
        apply_button.pack(pady=10)

    def apply_ford_fulkerson(graph, window, pos):
        # Ask user for source and sink nodes
        source = simpledialog.askinteger("Input", "Enter the source node:")
        sink = simpledialog.askinteger("Input", "Enter the sink node:")

        if source is not None and sink is not None and source in graph.nodes and sink in graph.nodes:
            # Apply Ford-Fulkerson algorithm
            flow_value, flow_dict = nx.maximum_flow(graph, source, sink)

            # Update the capacities with the flow values
            for u in flow_dict:
                for v in flow_dict[u]:
                    if graph.has_edge(u, v):
                        graph[u][v]['capacity'] -= flow_dict[u][v]

            # Create a frame to organize the graph and results side by side
            content_frame = ttk.Frame(window)
            content_frame.pack(fill="both", expand=True)

            # Left side: Updated graph
            graph_frame = ttk.Frame(content_frame)
            graph_frame.pack(side="left", fill="both", expand=True)

            fig, ax = plt.subplots(figsize=(6, 4))
            capacities = nx.get_edge_attributes(graph, 'capacity')
            nx.draw(graph, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500, font_size=10, ax=ax)
            nx.draw_networkx_edge_labels(graph, pos, edge_labels=capacities)
            ax.set_title("Graph with Updated Capacities after Ford-Fulkerson")

            canvas = FigureCanvasTkAgg(fig, master=graph_frame)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill="both", expand=True)
            canvas.draw()

            # Right side: Results
            result_frame = ttk.Frame(content_frame)
            result_frame.pack(side="right", fill="y", padx=10, pady=10)

            # Display the maximum flow result
            result_label = ttk.Label(result_frame, text=f"Maximum Flow: {flow_value}", font=("Helvetica", 14), foreground="blue")
            result_label.pack(pady=10)

            # Display flow details
            flow_details = "\n".join([f"{u} -> {v}: {flow}" for u in flow_dict for v, flow in flow_dict[u].items()])
            flow_text = f"Flow Details:\n{flow_details}"
            flow_label = ttk.Label(result_frame, text=flow_text, font=("Helvetica", 12), justify="left")
            flow_label.pack(pady=5, anchor="w")
        else:
            error_label = ttk.Label(window, text="Invalid source or sink node. Please try again.", font=("Helvetica", 12), foreground="red")
            error_label.pack(pady=10)

    open_graph_window()

def bellman_ford():
    
    def open_bellman_ford_graph_window():
        # Ask user for the number of nodes (sommets)
        num_nodes = simpledialog.askinteger("Input", "How many nodes do you want?", minvalue=1)
        if num_nodes is None:
            return

        # Generate a random directed graph with 50% probability of connection between nodes and random weights
        graph = nx.DiGraph()
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and random.random() < 0.5:  # 50% probability of connection
                    weight = random.randint(1, 10)  # Random weight between 1 and 10
                    graph.add_edge(i, j, weight=weight)

        # Create a new window
        graph_window = Toplevel(root)
        graph_window.title("Random Directed Graph")

        # Create a matplotlib figure
        fig, ax = plt.subplots(figsize=(6, 4))
        pos = nx.spring_layout(graph)
        weights = nx.get_edge_attributes(graph, 'weight')
        nx.draw(graph, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500, font_size=10, ax=ax, arrows=True)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=weights)
        ax.set_title("Random Directed Graph with Weights")

        # Embed the matplotlib figure in the tkinter window
        canvas = FigureCanvasTkAgg(fig, master=graph_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)
        canvas.draw()

        # Add an Apply button
        apply_button = ttk.Button(graph_window, text="Apply Bellman-Ford", command=lambda: apply_bellman_ford(graph, graph_window))
        apply_button.pack(pady=10)

    def apply_bellman_ford(graph, window):
        start_node = simpledialog.askinteger("Input", "Enter the start node:")
        
        if start_node is not None and start_node in graph.nodes:
            distances = {node: float('inf') for node in graph}
            distances[start_node] = 0
            previous_nodes = {node: None for node in graph}

            # Bellman-Ford algorithm
            for _ in range(len(graph) - 1):
                for u, v, data in graph.edges(data=True):
                    weight = data['weight']
                    if distances[u] + weight < distances[v]:
                        distances[v] = distances[u] + weight
                        previous_nodes[v] = u

            # Check for negative-weight cycles
            for u, v, data in graph.edges(data=True):
                weight = data['weight']
                if distances[u] + weight < distances[v]:
                    print("Graph contains a negative-weight cycle")
                    return

            # Create a frame for the table
            table_frame = ttk.Frame(window)
            table_frame.pack(fill="both", expand=True)

            # Create the table headers
            headers = ["Node", "Distance (KM)", "Path"]
            for col_num, header in enumerate(headers):
                header_label = ttk.Label(table_frame, text=header, font=("Helvetica", 12), borderwidth=2, relief="groove")
                header_label.grid(row=0, column=col_num, sticky="nsew")

            # Populate the table with distances and paths
            for row_num, (node, distance) in enumerate(distances.items(), start=1):
                path = get_path(previous_nodes, start_node, node)
                node_label = ttk.Label(table_frame, text=node, font=("Helvetica", 12), borderwidth=2, relief="groove")
                node_label.grid(row=row_num, column=0, sticky="nsew")
                distance_label = ttk.Label(table_frame, text=f"{distance} KM", font=("Helvetica", 12), borderwidth=2, relief="groove")
                distance_label.grid(row=row_num, column=1, sticky="nsew")
                path_label = ttk.Label(table_frame, text=" -> ".join(map(str, path)), font=("Helvetica", 12), borderwidth=2, relief="groove")
                path_label.grid(row=row_num, column=2, sticky="nsew")

            # Configure table columns to expand equally
            for col_num in range(len(headers)):
                table_frame.grid_columnconfigure(col_num, weight=1)

            # Configure table rows to expand equally
            for row_num in range(len(distances) + 1):
                table_frame.grid_rowconfigure(row_num, weight=1)

    def get_path(previous_nodes, start_node, target_node):
        path = []
        current_node = target_node
        while current_node is not None:
            path.append(current_node)
            current_node = previous_nodes[current_node]
        path.reverse()
        return path

    open_bellman_ford_graph_window()

def transportation_problem():
    # Create a new window for the transportation problem
    transport_window = Toplevel(root)
    transport_window.title("Transportation Problem")
    transport_window.state("zoomed")  # Set full screen mode
    
    class TransportationApp:
        def __init__(self, root):
            self.root = root

            # Create a frame for the table
            self.table_frame = ttk.Frame(root, padding=20)
            self.table_frame.pack(fill="both", expand=True)

            # Generate the table with random values
            self.generate_table()

            # Create button frame
            button_frame = ttk.Frame(root)
            button_frame.pack(fill="x", padx=10, pady=10)

            # Create buttons with ttkbootstrap styling
            self.north_west_button = ttk.Button(
                button_frame, 
                text="North-West Corner Method", 
                bootstyle=SUCCESS,
                command=self.north_west_corner_method
            )
            self.north_west_button.pack(side="left", padx=5)

            self.least_cost_button = ttk.Button(
                button_frame, 
                text="Least Cost Method", 
                bootstyle=SUCCESS,
                command=self.least_cost_method
            )
            self.least_cost_button.pack(side="left", padx=5)

            self.stepping_stone_button = ttk.Button(
                button_frame, 
                text="Stepping Stone Method", 
                bootstyle=SUCCESS,
                command=self.stepping_stone_method
            )
            self.stepping_stone_button.pack_forget()  # Initially hidden

            self.optimal_solution_label = ttk.Label(root, text="", font=("Helvetica", 12))
            self.optimal_solution_label.pack(pady=10)

        def generate_table(self):
            # Randomly generate the supply, demand, and cost matrix
            self.supply = [random.randint(20, 50) for _ in range(3)]
            self.demand = [random.randint(20, 50) for _ in range(3)]

            # Balance supply and demand
            total_supply = sum(self.supply)
            total_demand = sum(self.demand)
            
            # Ensure supply equals demand
            if total_supply > total_demand:
                self.demand.append(total_supply - total_demand)
                self.cost_matrix = np.random.randint(1, 10, size=(3, len(self.demand)))
            elif total_supply < total_demand:
                self.supply.append(total_demand - total_supply)
                self.cost_matrix = np.random.randint(1, 10, size=(len(self.supply), 3))
            else:
                self.cost_matrix = np.random.randint(1, 10, size=(3, 3))

            self.original_cost_matrix = self.cost_matrix.copy()  # Save original costs
            self.display_table(self.cost_matrix, self.supply, self.demand)

        def display_table(self, data, supply, demand):
            for widget in self.table_frame.winfo_children():
                widget.destroy()

            # Add headers
            headers = ["Usine\\Magasin"] + [f"Magasin {i + 1}" for i in range(len(demand))] + ["Capacité"]
            for col_num, header in enumerate(headers):
                header_label = ttk.Label(self.table_frame, text=header, font=("Helvetica", 12), borderwidth=2, relief="groove")
                header_label.grid(row=0, column=col_num, sticky="nsew")

            # Add rows with colors based on allocation
            max_value = np.max(data) if np.any(data) else 1
            for i, row in enumerate(data):
                row_label = ttk.Label(self.table_frame, text=f"Usine {i + 1}", font=("Helvetica", 12), borderwidth=2, relief="groove")
                row_label.grid(row=i + 1, column=0, sticky="nsew")

                for j, value in enumerate(row):
                    # Calculate color intensity based on value
                    intensity = int(200 * (1 - value/max_value)) if max_value > 0 else 255
                    color = f'#{intensity:02x}{intensity:02x}ff'  # Blue color with varying intensity
                    
                    cell_label = ttk.Label(
                        self.table_frame,
                        text=f"{value:.1f}",
                        font=("Helvetica", 12),
                        borderwidth=2,
                        relief="groove",
                        background=color if value > 0 else "white"
                    )
                    cell_label.grid(row=i + 1, column=j + 1, sticky="nsew")

                supply_label = ttk.Label(self.table_frame, text=supply[i], font=("Helvetica", 12), borderwidth=2, relief="groove")
                supply_label.grid(row=i + 1, column=len(row) + 1, sticky="nsew")

            # Add demand row
            demand_label = ttk.Label(self.table_frame, text="Demande", font=("Helvetica", 12), borderwidth=2, relief="groove")
            demand_label.grid(row=len(data) + 1, column=0, sticky="nsew")

            for j, demand_value in enumerate(demand):
                demand_cell = ttk.Label(self.table_frame, text=demand_value, font=("Helvetica", 12), borderwidth=2, relief="groove")
                demand_cell.grid(row=len(data) + 1, column=j + 1, sticky="nsew")

        def north_west_corner_method(self):
            allocation = np.zeros_like(self.cost_matrix)
            supply = self.supply[:]
            demand = self.demand[:]

            i, j = 0, 0
            while i < len(supply) and j < len(demand):
                allocated = min(supply[i], demand[j])
                allocation[i][j] = allocated
                supply[i] -= allocated
                demand[j] -= allocated

                if supply[i] == 0:
                    i += 1
                elif demand[j] == 0:
                    j += 1

            total_cost = np.sum(allocation * self.cost_matrix)
            self.north_west_total = total_cost
            self.north_west_allocation = allocation
            self.display_table(allocation, self.supply, self.demand)
            self.optimal_solution_label.config(text=f"North-West Corner Method Applied\nTotal Cost: {total_cost:.2f}")
            self.stepping_stone_button.pack(side="left", padx=5)

        def least_cost_method(self):
            allocation = np.zeros_like(self.cost_matrix)
            supply = self.supply[:]
            demand = self.demand[:]
            costs = self.cost_matrix.astype(float).copy()

            while np.any(supply) and np.any(demand):
                i, j = divmod(costs.argmin(), costs.shape[1])
                allocated = min(supply[i], demand[j])
                allocation[i][j] = allocated
                supply[i] -= allocated
                demand[j] -= allocated

                if supply[i] == 0:
                    costs[i, :] = np.inf
                if demand[j] == 0:
                    costs[:, j] = np.inf

            total_cost = np.sum(allocation * self.cost_matrix)
            self.least_cost_total = total_cost
            self.least_cost_allocation = allocation
            self.display_table(allocation, self.supply, self.demand)
            self.optimal_solution_label.config(text=f"Least Cost Method Applied\nTotal Cost: {total_cost:.2f}")
            self.stepping_stone_button.pack(side="left", padx=5)

        def find_entering_variable(self, allocation, u, v):
            m, n = allocation.shape
            min_cost = float('inf')
            entering_cell = None
            
            for i in range(m):
                for j in range(n):
                    if allocation[i, j] == 0:
                        reduced_cost = self.cost_matrix[i, j] - u[i] - v[j]
                        if reduced_cost < min_cost:
                            min_cost = reduced_cost
                            entering_cell = (i, j)
            
            return entering_cell, min_cost

        def compute_potentials(self, allocation):
            m, n = allocation.shape
            basic_cells = [(i, j) for i in range(m) for j in range(n) if allocation[i, j] > 0]
            
            # Initialize potentials
            u = [None] * m
            v = [None] * n
            u[0] = 0  # Set first row potential to 0
            
            # Compute potentials for basic cells
            changed = True
            while changed:
                changed = False
                for i, j in basic_cells:
                    if u[i] is not None and v[j] is None:
                        v[j] = self.cost_matrix[i, j] - u[i]
                        changed = True
                    elif u[i] is None and v[j] is not None:
                        u[i] = self.cost_matrix[i, j] - v[j]
                        changed = True
            
            # Fill any remaining None values
            u = [0 if x is None else x for x in u]
            v = [0 if x is None else x for x in v]
            
            return u, v

        def find_loop(self, allocation, start_pos):
            def get_next_cells(current_pos, direction='horizontal'):
                i, j = current_pos
                if direction == 'horizontal':
                    return [(i, col) for col in range(allocation.shape[1]) 
                           if (col != j and (allocation[i, col] > 0 or (i, col) == start_pos))]
                else:
                    return [(row, j) for row in range(allocation.shape[0]) 
                           if (row != i and (allocation[row, j] > 0 or (row, j) == start_pos))]

            def find_path(current_pos, visited, direction):
                if len(visited) > 2 and current_pos == start_pos:
                    return visited
                
                next_cells = get_next_cells(current_pos, direction)
                next_direction = 'vertical' if direction == 'horizontal' else 'horizontal'
                
                for next_pos in next_cells:
                    if len(visited) <= 2 or next_pos != start_pos:
                        if next_pos not in visited:
                            new_path = find_path(next_pos, visited + [next_pos], next_direction)
                            if new_path:
                                return new_path
                return None

            path = find_path(start_pos, [start_pos], 'horizontal')
            return path

        def stepping_stone_method(self):
            # Start with the best initial solution
            if hasattr(self, 'north_west_total') and hasattr(self, 'least_cost_total'):
                if self.north_west_total <= self.least_cost_total:
                    current_allocation = self.north_west_allocation.copy()
                else:
                    current_allocation = self.least_cost_allocation.copy()
            else:
                self.optimal_solution_label.config(text="Please run North-West or Least Cost method first!")
                return

            iteration = 0
            max_iterations = 100  # Prevent infinite loops
            improved = True

            while improved and iteration < max_iterations:
                improved = False
                iteration += 1

                # Compute potentials (u and v)
                u, v = self.compute_potentials(current_allocation)

                # Find entering variable
                entering_cell, min_reduced_cost = self.find_entering_variable(current_allocation, u, v)

                if min_reduced_cost >= -1e-10 or entering_cell is None:  # Solution is optimal
                    break

                # Find loop
                loop = self.find_loop(current_allocation, entering_cell)
                if not loop:
                    continue

                # Find maximum allowable change
                theta = float('inf')
                for i in range(1, len(loop), 2):  # Check negative positions
                    theta = min(theta, current_allocation[loop[i]])

                # Apply changes along the loop
                for i, pos in enumerate(loop):
                    if i % 2 == 0:  # Add theta
                        current_allocation[pos] += theta
                    else:  # Subtract theta
                        current_allocation[pos] -= theta

                improved = True

            # Calculate final cost
            final_cost = np.sum(current_allocation * self.cost_matrix)
            
            # Display results
            self.display_table(current_allocation, self.supply, self.demand)
            if iteration < max_iterations:
                self.optimal_solution_label.config(
                    text=f"Stepping Stone Method: Optimal Solution Found\n" +
                         f"Final Cost: {final_cost:.2f}"
                )
            else:
                self.optimal_solution_label.config(
                    text=f"Stepping Stone Method: Maximum iterations reached\n" +
                         f"Current Cost: {final_cost:.2f}"
                )

    app = TransportationApp(transport_window)

def dijkstra():
    
    def open_dijkstra_graph_window():
        # Ask user for the number of nodes (sommets)
        num_nodes = simpledialog.askinteger("Input", "How many nodes do you want?", minvalue=1)
        if num_nodes is None:
            return

        # Generate a random undirected graph with 50% probability of connection between nodes and random weights
        graph = nx.Graph()
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.random() < 0.5:  # 50% probability of connection
                    weight = random.randint(1, 10)  # Random weight between 1 and 10
                    graph.add_edge(i, j, weight=weight)

        # Create a new window
        graph_window = Toplevel(root)
        graph_window.title("Random Graph")

        # Create a matplotlib figure
        fig, ax = plt.subplots(figsize=(6, 4))
        pos = nx.spring_layout(graph)
        weights = nx.get_edge_attributes(graph, 'weight')
        nx.draw(graph, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500, font_size=10, ax=ax)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=weights)
        ax.set_title("Random Graph with Weights")

        # Embed the matplotlib figure in the tkinter window
        canvas = FigureCanvasTkAgg(fig, master=graph_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)
        canvas.draw()

        # Add an Apply button
        apply_button = ttk.Button(graph_window, text="Apply Dijkstra", command=lambda: apply_dijkstra(graph, graph_window))
        apply_button.pack(pady=10)

    def apply_dijkstra(graph, window):
        start_node = simpledialog.askinteger("Input", "Enter the start node:")

        if start_node is not None and start_node in graph.nodes:
            distances = {node: float('inf') for node in graph}
            distances[start_node] = 0
            previous_nodes = {node: None for node in graph}
            priority_queue = [(0, start_node)]

            while priority_queue:
                current_distance, current_node = heapq.heappop(priority_queue)

                if current_distance > distances[current_node]:
                    continue

                for neighbor in graph.neighbors(current_node):
                    weight = graph[current_node][neighbor]['weight']
                    distance = current_distance + weight

                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous_nodes[neighbor] = current_node
                        heapq.heappush(priority_queue, (distance, neighbor))

            # Create a frame for the table
            table_frame = ttk.Frame(window)
            table_frame.pack(fill="both", expand=True)

            # Create the table headers
            headers = ["Node", "Distance (KM)", "Path"]
            for col_num, header in enumerate(headers):
                header_label = ttk.Label(table_frame, text=header, font=("Helvetica", 12), borderwidth=2, relief="groove")
                header_label.grid(row=0, column=col_num, sticky="nsew")

            # Populate the table with distances and paths
            for row_num, (node, distance) in enumerate(distances.items(), start=1):
                path = get_path(previous_nodes, start_node, node)
                node_label = ttk.Label(table_frame, text=node, font=("Helvetica", 12), borderwidth=2, relief="groove")
                node_label.grid(row=row_num, column=0, sticky="nsew")
                distance_label = ttk.Label(table_frame, text=f"{distance} KM", font=("Helvetica", 12), borderwidth=2, relief="groove")
                distance_label.grid(row=row_num, column=1, sticky="nsew")
                path_label = ttk.Label(table_frame, text=" -> ".join(map(str, path)), font=("Helvetica", 12), borderwidth=2, relief="groove")
                path_label.grid(row=row_num, column=2, sticky="nsew")

            # Configure table columns to expand equally
            for col_num in range(len(headers)):
                table_frame.grid_columnconfigure(col_num, weight=1)

            # Configure table rows to expand equally
            for row_num in range(len(distances) + 1):
                table_frame.grid_rowconfigure(row_num, weight=1)

    def get_path(previous_nodes, start_node, target_node):
        path = []
        current_node = target_node
        while current_node is not None:
            path.append(current_node)
            current_node = previous_nodes[current_node]
        path.reverse()
        return path

    open_dijkstra_graph_window()
  
def kruskal():
    # Ask user for the number of nodes (sommets)
    num_nodes = simpledialog.askinteger("Input", "How many nodes do you want?", minvalue=1)
    if num_nodes is None:
        return

    # Generate a random undirected graph with 50% probability of connection and random weights
    graph = nx.Graph()
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < 0.5:  # 50% probability of connection
                weight = random.randint(1, 10)  # Random weight between 1 and 10
                graph.add_edge(i, j, weight=weight)

    # Create a new window
    graph_window = Toplevel(root)
    graph_window.title("Kruskal's Algorithm - Minimum Spanning Tree")

    # Apply Kruskal's algorithm and display the result
    apply_kruskal(graph, graph_window)


def apply_kruskal(graph, window):
    # Apply Kruskal's algorithm to find the Minimum Spanning Tree (MST)
    mst = nx.minimum_spanning_tree(graph, algorithm='kruskal')

    # Calculate the total weight of the MST
    total_weight = sum(d['weight'] for u, v, d in mst.edges(data=True))

    # Visualize the MST
    fig, ax = plt.subplots(figsize=(6, 4))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500, font_size=10, ax=ax)
    nx.draw_networkx_edges(graph, pos, edgelist=mst.edges(), edge_color="red", width=2)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, 'weight'))
    ax.set_title("Minimum Spanning Tree (Kruskal)")

    # Embed the matplotlib figure in the tkinter window
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=BOTH, expand=True)
    canvas.draw()

    # Add a label to display the total weight of the MST
    result_label = ttk.Label(
        window,
        text=f"Total weight of MST: {total_weight}DH",
        font=("Arial", 12),
        foreground="blue",
    )
    result_label.pack(pady=10)

def main_window():
    global root  # Make root globally accessible
    root = ttk.Window(themename="flatly")
    root.title("JEDRAOUI Kenza G9 4IIR")
    root.state("zoomed")  # Set full screen mode
    
    # Rest of your main_window code remains the same...

def welsh_powell():
    global root  # Access the global root variable
    
    # Ask user for the number of nodes (sommets)
    num_nodes = simpledialog.askinteger("Input", "How many sommets do you want?", minvalue=1)
    if num_nodes is None:
        return
        
    # Generate a random graph with 50% probability of connection between nodes
    graph = nx.erdos_renyi_graph(num_nodes, p=0.5)
    
    # Create a new window
    graph_window = Toplevel(root)
    graph_window.title("Welsh-Powell Colored Graph")
    graph_window.geometry("800x600")  # Set a reasonable default size
    
    # Make the window resizable
    graph_window.resizable(True, True)
    
    # Add a frame to contain the graph and labels
    main_frame = ttk.Frame(graph_window)
    main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
    
    # Apply Welsh-Powell algorithm
    apply_welsh_powell(graph, main_frame, num_nodes)

def apply_welsh_powell(graph, frame, num_nodes):
    # Apply Welsh-Powell algorithm
    # Step a: Find the degree of each vertex
    degrees = {node: degree for node, degree in graph.degree}
    
    # Step b: Sort the vertices in order of descending degrees
    sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
    
    # Step c and d: Color the graph
    color_map = {}
    current_color = 0
    
    for node in sorted_nodes:
        if node not in color_map:
            current_color += 1
            color_map[node] = current_color
            for neighbor in sorted_nodes:
                if neighbor not in color_map:
                    # Check adjacency with all previously colored nodes of the same color
                    can_color = all(
                        not graph.has_edge(neighbor, other)
                        for other in color_map
                        if color_map[other] == current_color
                    )
                    if can_color:
                        color_map[neighbor] = current_color

    # Create figure with a larger size
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(top=0.9)  # Adjust to make room for title
    
    # Get colors for nodes
    colors = [color_map[node] for node in graph.nodes]
    
    # Draw the graph with improved visual settings
    nx.draw(
        graph,
        with_labels=True,
        node_color=colors,
        edge_color="gray",
        node_size=500,
        font_size=10,
        font_weight='bold',
        cmap=plt.cm.rainbow,
        ax=ax,
        node_shape='o',
        width=2,
        alpha=0.9
    )
    
    ax.set_title("Graph Colored with Welsh-Powell Algorithm", pad=20, fontsize=14, fontweight='bold')
    
    # Create canvas and pack it into the frame
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=BOTH, expand=True)
    canvas.draw()
    
    # Calculate the range of X(G)
    num_colors_used = max(color_map.values())
    x1 = num_colors_used
    x2 = num_nodes
    
    # Create a styled frame for the result
    result_frame = ttk.Frame(frame)
    result_frame.pack(fill='x', pady=10)
    
    # Add a label to display the range of X(G) with improved styling
    result_label = ttk.Label(
        result_frame,
        text=f"Chromatic Number Range: {x1} ≤ X(G) ≤ {x2}",
        font=("Helvetica", 12, "bold"),
        foreground="#2c3e50",  # Dark blue-grey color
    )
    result_label.pack(pady=10)
    
    # Add additional information label
    info_label = ttk.Label(
        result_frame,
        text=f"Number of Nodes: {num_nodes} | Colors Used: {num_colors_used}",
        font=("Helvetica", 10),
        foreground="#7f8c8d"  # Subtle grey color
    )
    info_label.pack(pady=5)

# Function to display the main interface
def show_main_interface():
    intro_window.destroy()
    main_window()

# Main window setup
def main_window():
    global root  # Make root globally accessible
    root = ttk.Window(themename="flatly")
    root.title("JEDRAOUI Kenza G9 4IIR")
    root.state("zoomed")  # Set full screen mode
    
    # Set an icon for the title
    try:
        root.iconbitmap("logo-emsi.ico")
    except Exception as e:
        print(f"Icon not found: {e}")

    # Add the image at the top with rounded corners
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(script_dir, "emsi2.png")
        img = Image.open(img_path)
        # Create rounded corners
        img = img.resize((1500, 225))
        mask = Image.new('L', img.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle([(0, 0), img.size], radius=30, fill=255)
        output = Image.new('RGBA', img.size, (0, 0, 0, 0))
        output.paste(img, mask=mask)
        
        photo = ImageTk.PhotoImage(output)
        image_label = ttk.Label(root, image=photo, anchor="center")
        image_label.image = photo
        image_label.grid(row=0, column=0, columnspan=3, sticky="nsew")
    except Exception as e:
        print(f"Image not found: {e}")

    # Button names and their corresponding functions
    buttons_data = [
        ("Welsh-Powell", welsh_powell),
        ("Kruskal", kruskal),
        ("Dijkstra", dijkstra),
        ("Potentiel Métra", potentiel_metra),  
        ("Ford-Fulkerson", ford_fulkerson),
        ("Bellman-Ford", bellman_ford),
    ]

    # Create and place the buttons with pill shape styling
    style = ttk.Style()
    style.configure('Pill.TButton', borderwidth=0, focuscolor='none', radius=25)
    
    for idx, (name, command) in enumerate(buttons_data):
        button = ttk.Button(
            root, 
            text=name, 
            bootstyle=(SUCCESS, "pill"),  # Added pill style
            command=command
        )
        button.grid(row=(idx // 3) + 1, column=idx % 3, padx=20, pady=15, sticky="nsew")

    # Add the Transportation Problem button spanning 3 columns
    transportation_button = ttk.Button(
        root, 
        text="Transportation Problem", 
        bootstyle=(PRIMARY, "pill"),  # Added pill style
        command=transportation_problem
    )
    transportation_button.grid(row=len(buttons_data) // 3 + 1, column=0, columnspan=3, padx=20, pady=15, sticky="nsew")

    # Configure row and column weights
    root.rowconfigure(0, weight=1, minsize=225)
    rows = len(buttons_data) // 3 + 1
    for i in range(1, rows + 1):
        root.rowconfigure(i, weight=1)
    for j in range(3):
        root.columnconfigure(j, weight=1)

    root.mainloop()

# Initial display window
intro_window = ttk.Window(themename="flatly")
intro_window.title("Welcome")
intro_window.state("zoomed")

# Top frame for the image and text
top_frame = ttk.Frame(intro_window)
top_frame.pack(fill="both", expand=True)

# Add the image with rounded corners
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, "emsi2.png")
    img = Image.open(img_path)
    # Create rounded corners
    img = img.resize((1900, 350))
    mask = Image.new('L', img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([(0, 0), img.size], radius=30, fill=255)
    output = Image.new('RGBA', img.size, (0, 0, 0, 0))
    output.paste(img, mask=mask)
    
    photo = ImageTk.PhotoImage(output)
    img_label = ttk.Label(top_frame, image=photo, anchor="center")
    img_label.image = photo
    img_label.pack(pady=20)
except Exception as e:
    print(f"Image not found: {e}")

# Create a custom font style for the text
custom_font = font.Font(
    family="Montserrat",  # Modern, distinctive font
    size=16,
    weight="bold"
)

# Create a frame for the text with a subtle background
text_frame = ttk.Frame(top_frame, style='Custom.TFrame')
text_frame.pack(pady=20)

# Style for the frame
style = ttk.Style()
style.configure('Custom.TFrame', background='#f8f9fa')

# Add styled text labels
made_by_label = ttk.Label(
    text_frame, 
    text="Made by:",
    font=custom_font,
    foreground="#5ab884",  # Dark blue-grey color
    background="#f8f9fa"
)
made_by_label.pack(pady=(10, 5))

name_label = ttk.Label(
    text_frame,
    text="JEDRAOUI Kenza",
    font=custom_font,
    foreground="#163323",  # Accent color for the name
    background="#f8f9fa"
)
name_label.pack(pady=(0, 10))

direction_label = ttk.Label(
    text_frame,
    text="Under the direction of:",
    font=custom_font,
    foreground="#5ab884",
    background="#f8f9fa"
)
direction_label.pack(pady=(10, 5))

supervisor_label = ttk.Label(
    text_frame,
    text="Mouna El Mkhalet",
    font=custom_font,
    foreground="#163323",  # Accent color for the name
    background="#f8f9fa"
)
supervisor_label.pack(pady=(0, 10))

# Bottom frame for buttons
bottom_frame = ttk.Frame(intro_window)
bottom_frame.pack(fill="x", pady=20)

# Start button with pill shape
start_button = ttk.Button(
    bottom_frame, 
    text="Start Program", 
    bootstyle=("success-outline", "pill"),
    command=show_main_interface
)
start_button.pack(side="left", expand=True, padx=20, pady=20, ipadx=20, ipady=10)

# Quit button with pill shape
quit_button = ttk.Button(
    bottom_frame, 
    text="Quit", 
    bootstyle=("danger-outline", "pill"),
    command=intro_window.destroy
)
quit_button.pack(side="right", expand=True, padx=20, pady=20, ipadx=20, ipady=10)

# Run the intro window
intro_window.mainloop()