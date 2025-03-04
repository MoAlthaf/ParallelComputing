import random
import threading
import time
import sys
from queue import Queue

latest_temperatures = {}
temperature_averages = {}

temperature_queue = Queue()
lock = threading.Lock()

def simulate_sensor(sensor_id):
    while True:
        temperature = random.randint(15, 40)
        with lock:
            latest_temperatures[sensor_id] = temperature  
        temperature_queue.put((sensor_id, temperature))  
        time.sleep(1)  

def process_temperatures():
    sensor_data = {}  
    while True:
        sensor_id, temperature = temperature_queue.get()  
        with lock:
            if sensor_id not in sensor_data:
                sensor_data[sensor_id] = []
            sensor_data[sensor_id].append(temperature)  
            temperature_averages[sensor_id] = sum(sensor_data[sensor_id]) / len(sensor_data[sensor_id])
        update_display()  
        temperature_queue.task_done()

def initialize_display(sensor_count):
    print("\nCurrent Temperatures:\n")
    print("Latest Temperatures: ", end="")
    for i in range(sensor_count):
        print(f"Sensor {i}: --°C ", end="")
    print("\n")
    for i in range(sensor_count):
        print(f"Sensor {i} Average: {' ' * 50} --°C")

def update_display():
    with lock:
        sys.stdout.write("\033[F" * (len(latest_temperatures) * 2 + 2))  
        sys.stdout.flush()
        print("Latest Temperatures: ", end="")
        for sensor_id in latest_temperatures:
            print(f"Sensor {sensor_id}: {latest_temperatures[sensor_id]}°C ", end="")
        print("\n")
        for sensor_id in latest_temperatures:
            avg_temp = temperature_averages.get(sensor_id, "--")
            print(f"Sensor {sensor_id} Average: {' ' * 50} {avg_temp:}°C")

def main(sensor_count=3):
    initialize_display(sensor_count)
    threads = []
    for i in range(sensor_count):
        thread = threading.Thread(target=simulate_sensor, args=(i,), daemon=True)
        threads.append(thread)
        thread.start()
    processing_thread = threading.Thread(target=process_temperatures, daemon=True)
    threads.append(processing_thread)
    processing_thread.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting program...")

if __name__ == "__main__":
    main(sensor_count=3)
