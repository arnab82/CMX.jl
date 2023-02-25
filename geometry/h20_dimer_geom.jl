using FermiCG
using Printf
atoms = []
push!(atoms,Atom(1,"O", [0.279826717549,      0.373565416234,     -1.6083825210736]))
push!(atoms,Atom(2,"H", [0.279826717549 ,     1.126508122540,     -1.01735873050]))
push!(atoms,Atom(3,"H", [0.279826717549 ,    -0.379377292072,     -1.0173587330503]))
push!(atoms,Atom(4,"O", [0.407029145314,      0.965355983853,     -2.987218149039]))
push!(atoms,Atom(5,"H", [0.016501883017 ,     1.652782702915,     -2.447621259387]))
push!(atoms,Atom(6,"H", [0.016501883017,      0.277929266617,     -2.447621257061]))
println(atoms)
basis = "sto-3g"
mol = Molecule(0,1,atoms,basis)
io = open("traj_water_dimer.xyz", "w");
n_steps = 60    
step_size = .5
for R in 1:n_steps
    scale +=step_size
    println(scale)
    xyz = @sprintf("%5i\n\n", length(mol.atoms))
    tmp = []
    count=0
    for a in mol.atoms[1:3]
        push!(tmp, Atom(count+=1,"H",[a.xyz[1], a.xyz[2], a.xyz[3]]))
        xyz = xyz * @sprintf("%6s %24.16f %24.16f %24.16f \n", a.symbol, a.xyz[1], a.xyz[2], a.xyz[3])
    end
    count1=3
    for a in mol.atoms[4:6]
        push!(tmp, Atom(count+=1,"H",[a.xyz[1]*scale, a.xyz[2], a.xyz[3]]))
        xyz = xyz * @sprintf("%6s %24.16f %24.16f %24.16f \n", a.symbol, a.xyz[1]*scale, a.xyz[2], a.xyz[3])
    end
    display(tmp)
    pymol=Molecule(0,1,tmp,basis)
    println(xyz)
    write(io, xyz);
end
close(io)
