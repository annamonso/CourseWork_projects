import sys
import pandas as pd
import csv


##############################
### BACKTRACKING ALGORITHM ###
##############################

def load_data(driving_file, parks_file, zones_file):
    Distances = {}
    Parks = {}
    Zones = {}

    # --- 1. Distances ---
    with open(driving_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames[1:]  # columnas destino
        for row in reader:
            origin = row['STATE']
            Distances[origin] = {}
            for dest in headers:
                val = row[dest].strip()
                try:
                    distance = int(val)
                    # solo agregamos si es una distancia válida y no -1
                    if distance > 0:
                        Distances[origin][dest] = distance
                except ValueError:
                    continue

    # --- 2. Parks ---
    parks_df = pd.read_csv(parks_file)
    for state in parks_df.columns[1:]:
        val = parks_df.iloc[0][state]
        if pd.notna(val):
            Parks[state] = int(val)

    # --- 3. Zones ---
    zones_df = pd.read_csv(zones_file)
    for state in zones_df.columns[1:]:
        val = zones_df.iloc[0][state]
        if pd.notna(val):
            Zones[state] = f"Z{int(val)}"

    return Distances, Parks, Zones

def get_next_zone(current_zone):
    '''
        Return the next zone of the graph 
    '''
    number = int(current_zone[1:])
    next_zone = f"Z{number+1}"
    
    return next_zone 


def get_possible_values(current_state, current_zone, zones, distances):
    '''
        Returns all possible next states in the next zone that are directly connected to the current state (distance > 0)
        Sort alphabetically 
    '''
    if current_zone == "Z12":
        return []
    next_zone = get_next_zone(current_zone)
    next_states = []
    # Possible states 
    for state, zone in zones.items():
        if zone == next_zone:
            next_states.append(state)
        
    # Check if there's a road between the current state and the next_state 
    dist_from_state = distances.get(current_state)
    
    next_valid_states = list(set(next_states) & set(dist_from_state.keys()))

    
    return sorted(next_valid_states)


def backtrack(path, total_distance, total_parks, current_zone, target_parks, zones, parks, distances):
    '''
    Recursive backtracking search.
    Returns (path, total_distance, total_parks) if successful, or None if no path found.
    '''
    current_state = path[-1]

    # Check if assignment is complete
    if current_zone == "Z12" and total_parks >= int(target_parks):
        return path, total_distance, total_parks

    # Get the next zone
    next_zone = get_next_zone(current_zone)

    # Get all possible next states
    next_states = get_possible_values(current_state, current_zone, zones, distances)

    # Try each possible next state
    for next_state in next_states:
        if next_state not in path:  
            
            new_distance = total_distance + distances[current_state][next_state]
            new_parks = total_parks + parks[next_state]
            new_path = path + [next_state]

            # Recursive call
            result = backtrack(new_path, new_distance, new_parks, next_zone,
                               target_parks, zones, parks, distances)

            if result is not None:
                return result

    return None, None, None

        


def backtracking_search(initial_state, target_parks, zones, parks, distances):
    '''
    Sets up and starts the backtracking search.
    '''
    initial_zone = zones[initial_state]
    path = [initial_state]
    total_distance = 0
    total_parks = parks[initial_state]  

    path, total_distance, total_parks = backtrack(path, total_distance, total_parks,
                         initial_zone, target_parks, zones, parks, distances)
    return path, total_distance, total_parks


#############################
######## MAIN ###############
#############################

if __name__ == "__main__":
    PARKS_VISITED = 0 

    if len(sys.argv) == 3:
        initial_state = sys.argv[1]
        number_of_parks = int(sys.argv[2])
    else:
        print("ERROR: Not enough or too many input arguments.")
        print(f"length {len(sys.argv)}")
        sys.exit(1)
    print()
    print("Monso Rodriguez, Anna, A20653296 solution:")
    print(f"Initial state: {initial_state}")
    print(f"Minimum number of parks: {number_of_parks}")

    driving_file = 'driving2.csv'
    parks_file = 'parks.csv'
    zones_file = 'zones.csv'

    distances, parks, zones = load_data(driving_file, parks_file, zones_file)

    path, total_distance, total_parks = backtracking_search(initial_state, number_of_parks, zones, parks, distances)

    if path:
        print()
        print(f"Solution path: {path}")
        print(f"Number of states on a path: {len(path)}")
        print(f"Path cost: {total_distance}")
        print(f"Number of national parks visited: {total_parks}")
        print()
    else:
        print()
        print("Solution path: FAILURE: NO PATH FOUND")
        print("Number of states on a path: 0")
        print("Path cost: 0")
        print("Number of national parks visited: 0")
        print()