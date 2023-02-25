
function get_circle_coordinates(center_x, center_y,center_z ,radius, num_points)
    coordinates= []

    for i in 1:num_points
        angle = 2 * π * i / num_points
        x=center_x+0.0
        y=  center_y+ radius * cos(angle)
        z=  center_z+ radius * sin(angle)
        push!(coordinates,[x,y,z])
    end
    return coordinates
    
end
function get_stretched_geometry(coordinates,scale)
    new_coordinates=[]
    for i in 1:8
        angle = 2 * π * i / 8
        x1=0.0
        y1=coordinates[i][2][1]+scale* cos(angle)
        z1=coordinates[i][3][2]+scale* sin(angle)
        push!(new_coordinates,[x1,y1,z1])
    end
    return new_coordinates
end
# Example usage
center_x, center_y ,center_z= 0, 0,0
radius = 1.5
num_points = 12
coordinates= get_circle_coordinates(center_x, center_y, center_z,radius, num_points)
println("THE INITIAL H12 COORDINATES",coordinates)
deleteat!(coordinates,3)
deleteat!(coordinates,5)
deleteat!(coordinates,7)
deleteat!(coordinates,9)
display(coordinates)
println(typeof(coordinates))
println("THE FINAL H8 COORDINATES",coordinates)
n_steps = 40
step_size = .05
for R in 1:n_steps
    scale = 1+R*step_size
    coordinates= get_circle_coordinates(0.0,0.0,0.0,2,12)
    tmp=[]
    for i in coordinates
        push!(tmp,["H",(i[1],i[2],i[3])]) 
    end
    deleteat!(tmp,3)
    deleteat!(tmp,5)
    deleteat!(tmp,7)
    deleteat!(tmp,9)
    new_coordinates=get_stretched_geometry(tmp,scale)
    display(new_coordinates)
end 
