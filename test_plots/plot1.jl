
w=[]
for r in 1:70
    x=π/24+(r*π/250)
    x=x/π*180
    push!(w,x)
    println(x)
end
println(w)