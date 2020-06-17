# NIR-colorization
Near Infrared Image colorization
Implementation of [Infrared image colorization using S-shape network.](https://waseda.pure.elsevier.com/en/publications/infrared-image-colorization-using-a-s-shape-network)
Original structure of colorization process
<div align=center><img width="600" src="https://raw.githubusercontent.com/endrol/NIR-colorization/master/IMG/overview2.png"/></div>
<div align=left>

## add global structure
Idea was derived from [Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors for Automatic Image Colorization with Simultaneous Classification](http://iizuka.cs.tsukuba.ac.jp/projects/colorization/en/)  
<div align=center><img width="600" src="https://raw.githubusercontent.com/endrol/NIR-colorization/master/IMG/model.png"/></div>
<div align=left>
By utilizing this global feature fusions, I hope to get a more universal result

## experiment result
Infrared image colorization paper example
<div align=center><img width="600" src="https://raw.githubusercontent.com/endrol/NIR-colorization/master/IMG/compare0.png"/></div>
<div align=left>
Adding global feature fusion model, result from dataset [scene.](https://waseda.pure.elsevier.com/en/publications/infrared-image-colorization-using-a-s-shape-network)  
<center class="half">
    <img src="https://raw.githubusercontent.com/endrol/NIR-colorization/master/IMG/7.jpg" width="200"/><img src="https://raw.githubusercontent.com/endrol/NIR-colorization/master/IMG/43.jpg" width="200"/><img src="https://raw.githubusercontent.com/endrol/NIR-colorization/master/IMG/51.jpg" width="200"/>
</center>
<div align=left>
