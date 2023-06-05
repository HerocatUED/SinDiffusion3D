# SinDiffusion3D
diffusion model for single 3D mesh

# Progress
- ***triplane encoder-decoder*** Done (based on [NFD-tripane_decoder](https://github.com/JRyanShue/NFD/tree/main/nfd/triplane_decoder))
- ***diffusion model*** TODO
- ***oct-tree optimize*** TODO

#  Quick Start
1. install pytorch and other requirements ```pip install -r requirements.txt```
2. put xxx.obj file into 'dataset' folder
3. run command ```python main.py --shape_name=xxx``` Note: $shape\_name$should be the same as input mesh file. If the input mesh is StoneWall.obj, then $--shape\_name=StoneWall$ 
4. output file can be found in 'output' folder
