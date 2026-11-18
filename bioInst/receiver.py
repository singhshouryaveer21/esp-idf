import socket
import csv

UDP_IP = "0.0.0.0"
UDP_PORT = 5000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

filename = 'wireless_sleep_data.csv'
print(f"Listening for ESP32 data on port {UDP_PORT}...")
print(f"Saving incoming data to {filename}")
print("Press Ctrl+C to stop recording.")

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        'Timestamp', 'Piezo', 'SpO2_Percent', 'HeartRate_BPM',
        'Audio_RMS', 'Acc_X', 'Acc_Y', 'Acc_Z'
    ])

    try:
        while True:
            data, addr = sock.recvfrom(1024)
            decoded_data = data.decode('utf-8').strip()
            row = decoded_data.split(',')

            if len(row) == 8:
                writer.writerow(row)
                print(row)

    except KeyboardInterrupt:
        print(f"\nRecording safely stopped. Data saved to {filename}")