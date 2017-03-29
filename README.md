# Maintenance Prediction
  
Predicting software maintainability from various software metric.
<br>
Objective of this project is to predict the change in particular software over a period of time depending upon oo metrics. 

It involves analyzing various Machine Learning models like Linear Regression , GRNN , Ward Neural Network and NNEP GRNN .
<pre>
<h2>Using Feature Selction </h2>

Choosing best 3 components to represent 

<h2>Using PCA </h2>
Choosing first 3 components capturing around 89% of data functionality

eigenvalue	proportion	cumulative
  5.50169	  0.6113 	  0.6113 	-0.408size2-0.406nom-0.389rfc-0.363size1-0.36lcom...
  1.42584	  0.15843	  0.76973	0.662mpc-0.453dit+0.337size1+0.268wmc-0.22dac...
  1.08321	  0.12036	  0.89008	0.756dit+0.523mpc+0.224dac-0.201wmc-0.174lcom...
  0.52622	  0.05847	  0.94855	0.602lcom-0.4dac-0.399wmc+0.374rfc-0.326size1...
  0.30683	  0.03409	  0.98264	-0.527dac+0.469wmc+0.442dit-0.295mpc-0.279size2...

Eigenvectors
 V1	 V2	 V3	 V4	 V5	
-0.0701	-0.4532	 0.7561	-0.0162	 0.4417	dit
-0.0449	 0.6624	 0.523 	 0.2505	-0.295 	mpc
-0.389 	 0.1768	 0.0794	 0.3737	 0.0971	rfc
-0.3596	-0.1646	-0.1738	 0.6023	 0.1876	lcom
-0.3551	-0.2196	 0.224 	-0.4003	-0.5275	dac
-0.3516	 0.2676	-0.2009	-0.3991	 0.4694	wmc
-0.4056	-0.1697	-0.1441	 0.0916	-0.1889	nom
-0.4083	-0.1884	-0.0489	-0.0211	-0.2787	size2
-0.3626	 0.3367	 0.0678	-0.3258	 0.2475	size1

Making 3 new component using above eigenvectors
</pre>

<h3> Developed by: </h3> 
	RISHABH JAIN <br>
	TEJAN PREET SINGH
