r = 0.74
a = 2
push!(atoms,Atom(1,"H", [0, 0*a, 0*r]))
push!(atoms,Atom(2,"H", [0, 0*a, 1*r]))
push!(atoms,Atom(3,"H", [0, 1*a,2*r]))
push!(atoms,Atom(4,"H", [0, 1*a,3*r]))
push!(atoms,Atom(5,"H", [0, 2*a, 4*r]))
push!(atoms,Atom(6,"H", [0, 2*a, 5*r]))
push!(atoms,Atom(7,"H", [0, 3*a, 6*r]))
push!(atoms,Atom(8,"H", [0, 3*a, 7*r]))
push!(atoms,Atom(9,"H", [0, 4*a, 8*r]))
push!(atoms,Atom(10,"H",[0, 4*a, 9*r]))
push!(atoms,Atom(11,"H",[0, 5*a, 10*r]))
push!(atoms,Atom(12,"H",[0, 5*a, 11*r]))
println(atoms)


basis = "sto-3g"
mol = Molecule(0,1,atoms,basis)
n_steps=40
step_size = .05
io = open("traj.xyz", "w");
for R in 1:n_steps
    scale = 1+R*step_size
    xyz = @sprintf("%5i\n\n", length(mol.atoms))
    tmp = []
    count=0
    for a in mol.atoms
        push!(tmp, Atom(count+=1,"H",[a.xyz[1]*scale, a.xyz[2]*scale, a.xyz[3]*scale]))
        xyz = xyz*@sprintf("%6s %24.16f %24.16f %24.16f \n", a.symbol, a.xyz[1]*scale, a.xyz[2]*scale, a.xyz[3]*scale)
    end
    display(tmp)
    println(xyz)
    write(io,xyz);
end
close(io)
empty!(atoms)