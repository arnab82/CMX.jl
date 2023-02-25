import math
def get_circle_coordinates(center_x, center_y,center_z ,radius, num_points):
    coordinates_x = []
    coordinates_y = []
    coordinates_z = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x=center_x+0.0
        y =  center_y+ radius * math.cos(angle)
        z=  center_z+ radius * math.sin(angle)
        coordinates_x.append(x)
        coordinates_y.append(y)
        coordinates_z.append(z)
    return coordinates_x,coordinates_y,coordinates_z

# Example usage
center_x, center_y ,center_z= 0, 0,0
radius = 2
num_points = 12
coordinates_x,coordinates_y,coordinates_z= get_circle_coordinates(center_x, center_y, center_z,radius, num_points)
print(coordinates_x,coordinates_y)
n_steps = 40
step_size = .05
for R in range(n_steps):
    scale = 1+R*step_size
    coordinates_x,coordinates_y,coordinates_z= get_circle_coordinates(0.0,0.0,0.0,2*scale,12) 
    print(coordinates_x,coordinates_y,coordinates_z)   




'''H 0.0 2.0 0.0
H 0.0 1.73205 1.0
H 0.0 1.0 1.73205
H 0.0 0.0 2.0
H 0.0 -1.0 1.73205
H 0.0 -1.73205 1.0
H 0.0 -2.0 0.0
H 0.0 -1.73205 -1.0
H 0.0 -1.0 -1.73205
H 0.0 -0.0 -2.0
H 0.0 1.0 -1.73205
H 0.0 1.73205 -1.0'''