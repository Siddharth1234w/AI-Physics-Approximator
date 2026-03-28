import csv
import math
import random

def generate_projectile_data(num_samples=10000, filename="data/projectile_dataset.csv"):
    gravity = 9.81 
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # We added Initial_Height_m as a new feature
        writer.writerow(["Velocity_mps", "Angle_degrees", "Initial_Height_m", "Flight_Time_s", "Max_Height_m", "Distance_m"])
        
        for _ in range(num_samples):
            velocity = round(random.uniform(10.0, 100.0), 2)
            angle = round(random.uniform(5.0, 85.0), 2)
            height = round(random.uniform(0.0, 50.0), 2) # Launching from 0 to 50 meters high
            
            angle_rad = math.radians(angle)
            v_y = velocity * math.sin(angle_rad)
            v_x = velocity * math.cos(angle_rad)
            
            # Updated physics equations for a raised launch platform
            flight_time = round((v_y + math.sqrt(v_y**2 + 2 * gravity * height)) / gravity, 4)
            max_height = round(height + (v_y**2) / (2 * gravity), 4)
            distance = round(v_x * flight_time, 4)
            
            writer.writerow([velocity, angle, height, flight_time, max_height, distance])
            
    print(f"Success! {num_samples} records generated with the new Initial Height feature.")

if __name__ == "__main__":
    generate_projectile_data()