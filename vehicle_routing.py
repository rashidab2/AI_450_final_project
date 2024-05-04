import numpy as np

class VehicleRoutingProblem:
    def __init__(self, num_vehicles, depot, delivery_locations, vehicle_capacity):
        self.num_vehicles = num_vehicles
        self.depot = depot
        self.delivery_locations = delivery_locations
        self.vehicle_capacity = vehicle_capacity

    def calculate_distance(self, location1, location2):
        # Calculate distance between two locations (e.g., using Euclidean distance)
        return np.linalg.norm(np.array(location1) - np.array(location2))

    def calculate_savings(self):
        savings = []
        for i in range(len(self.delivery_locations)):
            for j in range(i+1, len(self.delivery_locations)):
                dist_depot_i = self.calculate_distance(self.depot, self.delivery_locations[i])
                dist_depot_j = self.calculate_distance(self.depot, self.delivery_locations[j])
                dist_ij = self.calculate_distance(self.delivery_locations[i], self.delivery_locations[j])
                savings.append((i, j, dist_depot_i + dist_depot_j - dist_ij))
        return sorted(savings, key=lambda x: x[2], reverse=True)
    
    def balance_routes(self, routes):
        # Flatten the list of routes to find all assigned locations
        assigned_locations = [loc for route in routes for loc in route]
        # Find all unassigned locations
        unassigned_locations = set(range(len(self.delivery_locations))) - set(assigned_locations)

        # Attempt to assign unassigned locations to existing routes, if capacity allows
        for loc in unassigned_locations:
            for route in routes:
                if len(route) < self.vehicle_capacity:
                    route.append(loc)
                    break
        average_locations = sum(len(route) for route in routes) // len(routes)
        for route in routes:
            while len(route) > average_locations + 1:
                for target_route in routes:
                    if len(target_route) < average_locations:
                        target_route.append(route.pop())
                        break
                else:
                    # If no route was found to take a location, stop trying to balance
                    break

        return routes

    def solve(self):
        savings = self.calculate_savings()
        routes = [[] for _ in range(self.num_vehicles)]
        used_indices = set()  # Keep track of which indices have been used
        for i, j, saving in savings:
            for route in routes:
                if i in used_indices or j in used_indices:
                    continue
                if len(route) + 2 <= self.vehicle_capacity:  # Check vehicle capacity
                    if len(route) == 0:
                        route.extend([i, j])
                        used_indices.update([i, j])
                    elif i == route[0]:
                        route.insert(0, j)
                        used_indices.add(j)
                    elif j == route[-1]:
                        route.append(i)
                        used_indices.add(i)
                    elif j == route[0]:
                        route.insert(0, i)
                        used_indices.add(i)
                    elif i == route[-1]:
                        route.append(j)
                        used_indices.add(j)
                # Break the inner loop if both i and j have been added to a route
                if i in used_indices and j in used_indices:
                    break
        routes = self.balance_routes(routes)
        # Optimize routes (e.g., using nearest neighbor algorithm
        optimized_routes = self.optimize_routes(routes)
        return optimized_routes

    def optimize_routes(self, routes):
        optimized_routes = []
        for route in routes:
            # Convert delivery location indices to distance matrix indices
            route_indices = [self.depot] + route + [self.depot]  # Include depot as start and end
            # Optimizing the route using Tabu Search algorithm
            optimized_route_indices = self.tabu_search(route_indices)
            # Convert indices back to delivery locations
            optimized_route = [self.delivery_locations[i-1] for i in optimized_route_indices[1:-1]]  # Exclude depot
            optimized_routes.append(optimized_route)
        return optimized_routes
    def tabu_search(self, initial_route, iterations=100, tabu_tenure=10):
        best_route = initial_route
        best_cost = self.calculate_route_cost(best_route)
        tabu_list = []
        current_route = best_route
        current_cost = best_cost
        for _ in range(iterations):
            neighborhood = self.get_neighborhood(current_route)
            if not neighborhood:  # Check if neighborhood is empty
                break  # No neighbors to explore, exit the loop

            neighborhood_costs = [self.calculate_route_cost(neighbor) for neighbor in neighborhood]
            best_neighbor_index = np.argmin(neighborhood_costs)
            best_neighbor = neighborhood[best_neighbor_index]

            if neighborhood_costs[best_neighbor_index] < best_cost and best_neighbor not in tabu_list:
                best_route = best_neighbor
                best_cost = neighborhood_costs[best_neighbor_index]

            current_route = best_neighbor
            current_cost = neighborhood_costs[best_neighbor_index]
            tabu_list.append(current_route)
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)
        return best_route

    def calculate_route_cost(self, route):
        total_cost = 0
        for i in range(len(route) - 1):
            from_location = route[i]
            to_location = route[i + 1]
            total_cost += self.calculate_distance(from_location, to_location)
        return total_cost
    def get_neighborhood(self, route):
        neighborhood = []
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                new_route = route[:i] + route[i:j][::-1] + route[j:]
                neighborhood.append(new_route)
        return neighborhood
    
    def solve_with_branch_and_bound(self):
        self.best_routes = None
        self.best_cost = float('inf')
        self.current_routes = [[] for _ in range(self.num_vehicles)]
        self.branch_and_bound(0, set(range(len(self.delivery_locations))), 0,)
        return self.best_routes

    def branch_and_bound(self, vehicle_index, remaining_locations, current_cost,):
        # Calculate lower bound for the remaining locations
        lower_bound = self.calculate_lower_bound(remaining_locations, current_cost)
        # If the lower bound is higher than the best cost, prune this branch
        if lower_bound >= self.best_cost:
            return
        # If all locations are assigned, update the best solution if needed
        if not remaining_locations:
            if current_cost < self.best_cost:
                self.best_cost = current_cost
                self.best_routes = [list(route) for route in self.current_routes]
            return
        # If all vehicles are used, this branch cannot provide a better solution
        if vehicle_index >= self.num_vehicles:
            return
        for location_index in remaining_locations:
            if len(self.current_routes[vehicle_index]) < self.vehicle_capacity:
                # Calculate the cost of adding the current location to the route
                added_cost = self.calculate_added_cost(vehicle_index, location_index)
                # Add the location to the current route
                self.current_routes[vehicle_index].append(self.delivery_locations[location_index])
                # Recursively apply branch and bound
                self.branch_and_bound(vehicle_index, remaining_locations - {location_index}, current_cost + added_cost)
                # Backtrack
                self.current_routes[vehicle_index].pop()

        # Try the next vehicle
        if len(self.current_routes[vehicle_index]) == 0:
            self.branch_and_bound(vehicle_index + 1, remaining_locations, current_cost)

    def calculate_lower_bound(self, remaining_locations, current_cost):
        lb = current_cost
        for loc_index in remaining_locations:
            distance_to_depot = self.calculate_distance(self.depot, self.delivery_locations[loc_index])
            lb += 2 * distance_to_depot  # go to the location and back to the depot
        return lb

    def calculate_added_cost(self, vehicle_index, location_index):
        # Calculate the additional cost incurred by adding a new location to the current route
        if len(self.current_routes[vehicle_index]) == 0:
            return self.calculate_distance(self.depot, self.delivery_locations[location_index])
        else:
            last_location = self.current_routes[vehicle_index][-1]
            return self.calculate_distance(last_location, self.delivery_locations[location_index])



# Example usage
num_vehicles = 4
delivery_locations = [(228, 0),(912, 0),
                      (0, 80),(114, 80),(570, 160),
                      (798, 160),(342, 240),(684, 240),
                      (570, 400),(912, 400),(114, 480),  
                      (228, 480),(342, 560),(684, 560),
                      (0, 640),(798, 640)]  
depot = (456, 320)
vehicle_capacity = 10

vrp = VehicleRoutingProblem(num_vehicles, depot, delivery_locations, vehicle_capacity)
optimized_routes = vrp.solve()
print("Tabu Optimized Routes:")
for i, route in enumerate(optimized_routes):
    # Find indices for the locations in the route
    print(route)
    route_indices = [0]  # Start with the depot index
    for loc in route:
        # Find the index of the location in the delivery_locations list
        index = vrp.delivery_locations.index(loc)
        route_indices.append(index+1)  # Add 1 because indices are 0-based and we want to start from 1
    route_indices.append(0)  # End with the depot index

    # Convert the route indices to a string representation
    route_str = ' -> '.join(str(index) for index in route_indices)
    # Calculate the total distance of the route using the indices
    route_locations = [vrp.depot] + [vrp.delivery_locations[index-1] for index in route_indices[1:-1]] + [vrp.depot]
    route_distance = vrp.calculate_route_cost(route_locations)
    # Print the route information using indices and the calculated distance
    print(f"Route for vehicle {i + 1}:")
    print(route_str)
    print(f"Distance of the route: {route_distance}m\n")

vrp2 = VehicleRoutingProblem(num_vehicles, depot, delivery_locations, vehicle_capacity)
back_optimized_routes = vrp2.solve_with_branch_and_bound()
print("Backtracking Optimized Routes:")
for i, route in enumerate(back_optimized_routes):
    # Find indices for the locations in the route
    print(route)
    route_indices = [0]  # Start with the depot index
    for loc in route:
        # Find the index of the location in the delivery_locations list
        index = vrp2.delivery_locations.index(loc)
        route_indices.append(index+1)  # Add 1 because indices are 0-based and we want to start from 1
    route_indices.append(0)  # End with the depot index

    # Convert the route indices to a string representation
    route_str = ' -> '.join(str(index) for index in route_indices)
    # Calculate the total distance of the route using the indices
    route_locations = [vrp2.depot] + [vrp2.delivery_locations[index-1] for index in route_indices[1:-1]] + [vrp2.depot]
    route_distance = vrp2.calculate_route_cost(route_locations)
    # Print the route information using indices and the calculated distance
    print(f"Route for vehicle {i + 1}:")
    print(route_str)
    print(f"Distance of the route: {route_distance}m\n")
