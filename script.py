from serial.tools import list_ports

a = list_ports.comports()

for i,j,k in a:
	print(i,j,k)
