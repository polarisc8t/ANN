/***************************************************
 ANN PROJECT BY AKZCHEUNG
***************************************************/


#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "nnlayer.h"

using namespace std;

struct NNType
{
	int NumLayers,		// Number of layers in NN
		 epoch,			// Epoch training size
		 dual_epoch,    // dual epoch training size
		 bias,			// bias flag  (if 1, bias exists)
		 feedforward,  // feedforward (1=true, 0=no feedforward)
		 recurrency; 	// flag to allow recurrent connections

	NNLayerType LayerInfo[MAX_LAYER];	// Layer info for each layer

	double InScale[MAX_PE], 	// The NN input and output
		   OutScale[MAX_PE],	// scaling vectors
		   InOffset[MAX_PE],	// Input offset
		   OutOffset[MAX_PE],	// Output offset
		   InRange[MAX_PE][2],   // Ranges for Input
		   OutRange[MAX_PE][2];  // and Output (lower, upper)

	double initpw, initnw,   // init. weight range (pos,neg)
		   maxpw, maxnw;    // max weight range

};

// A weight type is used to construct a general NN weight
// matrix. 
struct weight
{
	double delta_w, w;	// delta_w is the eventual update to w.
	int active;				// the active flag means the connection is
							// active and can be updated if true. Else
							// the connection is skipped (w=0 always)
	// More weight parameters here
};

//***************
// NN class
//***************

class NN
{
	public:
		NN();						// Default Constructor
		NN(NNType);					// Constructor
		NN(char*);					// Constructor (init from file)
        void InitNN(char*);     // InitNN from a file (non-constructor)
		void InitWeights();		// Random Init of Weights
		void LoadWeights(char*);// Load weights from file
//		void ImportWeights(char*); // Imports NN weights from a NeuralWorks .WTS format
		void SaveWeights(char*);// Save weights to file
//		void ExportWeights(char*); // Exports NN weights to NeuralWorks .WTS format
		void SetInput(double[]);// Set the Input vector to NN
		void Evaluate();			// Determine new network state
		void DualPartial();		// Extract NN partial derivatives
		void DispPartial();		// Displays the Partial Derivs Matrix
		void DualBackProp(double[]);		// Perform backprop training
											// given an error vector
		void Dual2BackProp(double [MAX_PE][MAX_PE]); // trains the dual NN
		                          // given a desired derivatives matrix
		void DualTrain(double[]); // train the dual network with an error
        void WeightUpdate();    // Update the entire weight matrix

		// The overall NN weight matrix (memory hog!)
		weight W[MAX_LAYER][MAX_LAYER][MAX_PE][MAX_PE];

		double PartialMatrix[MAX_PE][MAX_PE];	// Partial Deriv Matrix

		double Input[MAX_PE],	// input vector (plus bias)
				 Output[MAX_PE];  // output of vector

		NNLayer Layer[MAX_LAYER];	// Allocate our NNLayers

		int NumLayers, // Store NumLayers here for ease
			NumIn, NumOut,	// Record number inputs and number of outputs
			dual_count,   // keeps track of the number of dual training iterations.
			count;		// This keeps track of the number of training iterations
						// When this equals the epoch size, the weights are updated

		NNType Type;	// Make a copy of the current NN setup

};

// NN-CLASS FUNCTION DEFINITIONS

NN::NN(): count(0),dual_count(0) // Default Constructor
{
   // Really nothing is done here, not even zeroing-out variables
   // except the counter. 

} // End Default Constructor

// Constructor 1
NN::NN(NNType _Type): NumLayers(_Type.NumLayers), Type(_Type),
	count(0), dual_count(0), NumOut(_Type.LayerInfo[NumLayers-1].NumPE)
// We initialize some member data and init our Layer objects
{
	int i;

	Type.LayerInfo[0].NumPE++;			// Make room for a bias input
	NumIn = Type.LayerInfo[0].NumPE;

	// Init each layer
	for(i=0;i<NumLayers;++i)
		Layer[i].InitNNLayer(Type.LayerInfo[i]);

	// We can set the bias element (which is the 0th element of
	// of the input vector) to BIAS.
	Input[0] = Type.bias;  // For normal bias, set "bias=1"

    InitWeights();

} // End NN(NNType) Constructor

// File-based NN creation
NN::NN(char* nnfile) : count(0), dual_count(0)
{
   // Since we already have an InitNN(Filename) function, we can
   // call it from here...

   InitNN(nnfile);

} // End NN(File)

// InitNN - File based non-constructor initializer.
void NN::InitNN(char* nnfile)
{
	FILE *nn_file = fopen(nnfile, "r");	// Open the file to read NN params

    char weight_file[80];

	int i,w_file_flag;

	cout << "\n Opening NN File: " << nnfile;

	if(!nn_file)
	{
		cout << "\n Couldn't Open NN file: " << nnfile << "\n";
		exit(1);
	}

	// File is open.. grab the NN data
	fscanf(nn_file,"NumLayers: %i\n", &Type.NumLayers);
	NumLayers = Type.NumLayers;
	fscanf(nn_file,"Epoch: %i\n", &Type.epoch);
	fscanf(nn_file,"Bias: %i\n", &Type.bias);
	fscanf(nn_file,"FeedForward: %i\n", &Type.feedforward);
	fscanf(nn_file,"Recurrency: %i\n", &Type.recurrency);
	fscanf(nn_file,"Initial Weight Range: %lf %lf\n",&Type.initpw,&Type.initnw);
	fscanf(nn_file,"Max Weight Range: %lf %lf\n",&Type.maxpw, &Type.maxnw);

	cout << "\n NumLayers: " << NumLayers;
	cout << "\n Epoch: " << Type.epoch;
	cout << "\n Bias: " << Type.bias;
	cout << "\n FeedForward: " << Type.feedforward;
	cout << "\n Recurrency: "<< Type.recurrency << "\n";

	Input[0] = Type.bias;  // For normal bias, set "bias=1"

	// Read in NumLayers worth of layer data
	for(i=0;i<NumLayers;++i)
	{
		fscanf(nn_file, "<><><>\n");
		fscanf(nn_file,"NumPE: %i\n", &Type.LayerInfo[i].NumPE);
		// cout << "\n" << Type.LayerInfo[i].NumPE;
		fscanf(nn_file,"ActType: %i\n", &Type.LayerInfo[i].ActType);
		fscanf(nn_file,"OutType: %i\n", &Type.LayerInfo[i].OutType);
		fscanf(nn_file,"F'Offset: %lf\n", &Type.LayerInfo[i].DerivOff);
		fscanf(nn_file,"Beta: %lf\n", &Type.LayerInfo[i].Beta);
		fscanf(nn_file,"Momentum: %lf\n", &Type.LayerInfo[i].Momentum);

		// cout << Type.LayerInfo[i].NumPE << " ";
		// Init the layer!
		if(i==0)
			Type.LayerInfo[0].NumPE++;		// Make room for a bias input

		Layer[i].InitNNLayer(Type.LayerInfo[i]);
	}


	NumIn = Type.LayerInfo[0].NumPE;
	NumOut = Type.LayerInfo[NumLayers-1].NumPE;

	Type.dual_epoch = NumOut*Type.epoch;  // this is a leap!

	// Finally, read the scaling and offset information from the ranges
	fscanf(nn_file,"InRange:\n");
	for(i=0;i<NumIn-1;++i)
	{
		fscanf(nn_file,"%lf %lf\n", &Type.InRange[i][0], &Type.InRange[i][1]);
		Type.InScale[i] = (Type.InRange[i][1]-Type.InRange[i][0])/2.0;
		Type.InOffset[i] = -(Type.InScale[i]+Type.InRange[i][0]);
		cout << "\n Inscale["<<i<<"] = "<<Type.InScale[i];
		cout << "\n Inoffset["<<i<<"] = "<<Type.InOffset[i];
	}

	fscanf(nn_file,"OutRange:\n");
	for(i=0;i<NumOut;++i)
	{
		fscanf(nn_file,"%lf %lf\n", &Type.OutRange[i][0], &Type.OutRange[i][1]);
		if(Layer[NumLayers-1].LayerType.ActType==0) // For Linear Output
		{
			Type.OutScale[i] = 1.0;
			Type.OutOffset[i] = 0.0;
		}
		else if(Layer[NumLayers-1].LayerType.ActType==1
			|| Layer[NumLayers-1].LayerType.ActType==3) // For tanh/sin Outputs
		{
			Type.OutScale[i] = (Type.OutRange[i][1]-Type.OutRange[i][0])/2.0;
			Type.OutOffset[i] = (Type.OutScale[i]+Type.OutRange[i][0]);
		}
		else if(Layer[NumLayers-1].LayerType.ActType==2
			    || Layer[NumLayers-1].LayerType.ActType==4) //For sigmoid/gauss Output
		{
			Type.OutScale[i] = (Type.OutRange[i][1]-Type.OutRange[i][0]);
			Type.OutOffset[i] = Type.OutRange[i][0];
		}
		cout << "\n Outscale["<<i<<"] = "<<Type.OutScale[i];
		cout << "\n Outoffset["<<i<<"] = "<<Type.OutOffset[i];
	}

   // If the NN has a pre-defined weight matrix, let's read it.

   fscanf(nn_file,"Use Weight File: %i\n",&w_file_flag);
   if(w_file_flag)
   {
      fscanf(nn_file,"Weight File: %s\n",weight_file);
      LoadWeights(weight_file);
   }
   else  // Randomize the weights
   {
	 cout << "\n Not using weight file...Initializing Weights";
	 InitWeights();
   }


	// That's it..

	fclose(nn_file);

} // End InitNN(File)

// InitWeights()
void NN::InitWeights()
// This function randomly initializes the weights in the weight
// matrix.
{
	// Prepare random number generator
	srand( (unsigned)time( NULL ) );

	int i,j,k,l;

	// i=FROM LAYER | j=FROM LAYER
	// k=FROM PE   | l=TO PE

	for(i=0;i<MAX_LAYER;++i)         // This will randomly init
		for(j=0;j<MAX_LAYER;++j)      // each possible connection
			for(k=0;k<MAX_PE;++k)      // between (initpw,initnw) and
				for(l=0;l<MAX_PE;++l)   // zeros-out the delta_w's.
				{
					W[i][j][k][l].w = double(rand()%
						int((Type.initpw-Type.initnw)*1000.0)+
						(Type.initnw*1000.0))/1000.0;

					W[i][j][k][l].delta_w = 0.0;
					W[i][j][k][l].active = 1;

					// Next, we'll check to see if it should be
					// active or not based on Bias, Feedforward, and
					// Recurrency.
					if(k>=Layer[i].NumPE)
						W[i][j][k][l].active = 0;

					// Feedforward check: If j>(i+1), we are feedforwarding
					if(j>(i+1)) // We are assessing a feedforward connection
						W[i][j][k][l].active = Type.feedforward;

					// Bias Check: W[0][X][0][Y].active = bias
					if(i==0 && k==0) // We are assessing a bias connection
						W[i][j][k][l].active = Type.bias;

					// Recurrency check: If j<=i, we are doing recurrency
					if(j<=i)	// We are assessing a recurrent connection
						W[i][j][k][l].active = Type.recurrency;
				}

	for(i=0;i<MAX_PE;++i)	// Zero-out the Partial Deriv Matrix
		for(j=0;j<MAX_PE;++j)
			PartialMatrix[i][j] = 0.0;

} // End of InitWeights

// LOAD AND SAVE

// SaveWeights
void NN::SaveWeights(char* filename)
// THis function outputs the current NN weight matrix to a file.
{
	int i,j,k,l;

	FILE *w_out = fopen(filename, "w");

	for(i=0;i<NumLayers;++i)      // i=from layer
	{
		for(j=0;j<NumLayers;++j)  // j - from layer
		{
			for(k=0;k<Layer[i].NumPE;++k)    // k - from pe
				for(l=0;l<Layer[j].NumPE;++l)	// l - to pe
					if(W[i][j][k][l].active)
						fprintf(w_out,"W[%d][%d][%d][%d]=%lf\n",i,j,k,l,W[i][j][k][l].w);
		}
	}

	fprintf(w_out,"W[0][0][0][0]=0.0\n");	// Tag the end (this connection shouldn't exist!

	fclose(w_out);

} // End SaveWeights

// LoadWeights
void NN::LoadWeights(char* filename)
// This loads the weights from a file in the format as saved by SaveWeights
{
	int i,j,k,l;
	double w;

	FILE* infile = fopen(filename, "r");

	InitWeights();	// Reset Everything

	while(1)
	{
		fscanf(infile,"W[%d][%d][%d][%d]=%lf\n",&i,&j,&k,&l,&w);
		if(i==0 && j==0 && k==0 && l==0 && w==0.0)	// We're at the end!
			break;

		W[i][j][k][l].active=1;
		W[i][j][k][l].w=w;
		W[i][j][k][l].delta_w=0.0;
	}

	// Done loading!

	fclose(infile);

} // End LoadWeights

// SetInput
void NN::SetInput(double inp[])
// This function sets the input vector to the NN
{
	int i;

	for(i=0;i<NumIn;++i)    // input[0] is the bias value
		Input[i+1]=inp[i];  // so input[1..MAX_PE-1] -> inp[0..MAX_PE-2]
					 // This also means the input layer can only
					 // have MAX_PE-1 max inputs. :p 

	for(i=NumIn;i<MAX_PE;++i)	// Let's zero out remaining inputs
		Input[i]=0.0;

} // END SetInput

// Evaluate()
void NN::Evaluate()
// This determines the output of the NN
// Supports Feedforwarding and a bias
// Currently no recurrency support
{
	int i,j,k,l,p;

	double temp[MAX_PE];		// temp sum vector

	for(i=1;i<NumIn;++i)	// Note, Input[0] = bias
		Input[i]=(Input[i]+Type.InOffset[i-1])/Type.InScale[i-1];

	// Set the input to the input layer (layer[0])
	Layer[0].SetInput(Input);
	Layer[0].Evaluate();	// generate output of input layer

//	cout << "\n Evaluating Network...";
	// Next we evaluate the network
	for(i=1;i<NumLayers;++i)	// i = "to" layer
	{
		for(k=0;k<MAX_PE;++k)
			temp[k]=0.0;

		for(j=(Type.feedforward||Type.bias?0:i-1);j<i;++j) // j = "from" layer
		{
			for(p=0;p<Layer[i].NumPE;++p)	// p = "to" PE of "to" layer
				for(l=0;l<Layer[j].NumPE;++l)	// l = "from" PE of "from" layer
				// Sum up all the signals going into layer i if the
				// particular connection is active.
					if(W[j][i][l][p].active)
						temp[p] += Layer[j].Output[l]*W[j][i][l][p].w;
		}

		/* cout << "\n To layer: " << i << " => ";
		for(j=0;j<MAX_PE;++j)
			cout << temp[j] << ",";
		*/

		Layer[i].SetInput(temp);	// Set the input to the "to" layer (i)
		Layer[i].Evaluate();		// Generate the output of layer i.

		// Now we are ready to evaluate the next layer
	}

//	cout << "\n Scaling Output...";
	// We have the final output...scale it and store it as output
	// Output = Output*Scale+Offset
	for(j=0;j<NumOut;++j)
		Output[j]=Layer[NumLayers-1].Output[j]*Type.OutScale[j]
      +Type.OutOffset[j];

} // END Evaluate

// DualPartial
void NN::DualPartial()
// This function fills the Partial Deriv matrix
{
	double test[MAX_PE], sum[MAX_PE];

	int i=0,j=0,k=0,l=0,m=0;

	for(m=0;m<NumOut;++m)	// m-for each output component
	{
//		cout << "\n";
		for(i=0;i<NumOut;++i)			// Form our component vector to backprop
		{

			test[i]=(i==m)?Type.OutScale[i]:0.0;
//			cout << test[i];
		}

		Layer[NumLayers-1].SetPartialOutput(test);  // Set output to component vector
		Layer[NumLayers-1].DualPartial();	 // Do a backprop at output layer

		for(i=NumLayers-2;i>=0;--i)	// i = to layer
		{
			for(j=0;j<MAX_PE;++j)	// Zero out sum vector
				sum[j]=0.0;

			for(j=(Type.feedforward)?NumLayers-1:i+1;j>i;--j) // j = from layer
				for(k=0;k<Layer[i].NumPE;++k)	// k - to PE
				{
					for(l=0;l<Layer[j].NumPE;++l)	// l - from PE
					{
						if(W[i][j][k][l].active)
							sum[k]+=Layer[j].BackPartial[l]*W[i][j][k][l].w;
					}
					//cout << "\nSum["<<i<<k<<j<<"]:="<<sum[k];

				}
			// Here we can now set layer i's output and backprop again
			Layer[i].SetPartialOutput(sum);
			Layer[i].DualPartial();
	/*		for(j=0;j<Layer[i].NumPE;++j)
				cout << "\n" << Layer[i].BackPartial[j];
	*/
		}

      for(i=1;i<NumIn;++i)
         PartialMatrix[m][i-1] = Layer[0].BackPartial[i]/Type.InScale[i-1];

	}
	// At this point the PartialMatrix is filled.. we're done!

} // End DualPartial

// Display Partial
void NN::DispPartial()
{
	int i,j;

   cout << "\n Partial Derivatives Matrix:";
   cout << "\n Rows=Output : Columns=Input\n";

	for(i=0;i<NumOut;++i)
	{
      for(j=0;j<NumIn-1;++j)
			cout << " " << PartialMatrix[i][j];
		cout << "\n";
	}
} // End DispPartial

// DualBackProp
void NN::DualBackProp(double e[])
{

	int i,j,k,l;

	double sum[MAX_PE];

	// First, we increment our counter... If the counter equals the epoch
	// size then the weights are updated (w=w+delta_w)
	count++;

	// Next we need to convert our error back into unit-less values
	// by performing the inverse scaling and offsetting:

	for(j=0;j<NumOut;++j)
		e[j]=e[j]*Type.OutScale[j];

	// Now we can backpropagate our new unitless error vector

	Layer[NumLayers-1].SetOutput(e);  // Set output to component vector
	Layer[NumLayers-1].DualBackProp();	 // Do a backprop at output layer

	for(i=NumLayers-2;i>=0;--i)	// i = to layer
	{
		for(j=0;j<MAX_PE;++j)	// Zero out sum vector
			sum[j]=0.0;

		for(j=(Type.bias||Type.feedforward)?NumLayers-1:i+1;j>i;--j) // j = from layer
		{
			for(k=0;k<Layer[i].NumPE;++k)	// k - to PE
			{
				for(l=0;l<Layer[j].NumPE;++l)	// l - from PE
				{
					if(W[i][j][k][l].active)
					{
						// Determine backpropagated error
						sum[k]+=Layer[j].BackProp[l]*W[i][j][k][l].w;

						// Calculate the delta_w for each active connection
						// delta_w+=(Beta/epoch)*x*s+Alpha*delta_w
						// Added Beta/epoch term!

						W[i][j][k][l].delta_w = Layer[j].LayerType.Beta
                          /(double(Type.epoch))*Layer[i].Output[k]
                          *Layer[j].BackProp[l]+Layer[j].LayerType.Momentum
                          *W[i][j][k][l].delta_w;
					    //cout << "\nW"<<i<<j<<k<<l<<".dw="<<W[i][j][k][l].delta_w;

						if(count==Type.epoch)	// Is it time to update w?
						{
							// Update weight with delta_w
							W[i][j][k][l].w+=W[i][j][k][l].delta_w;
							// Are we out of weight range???
							if(W[i][j][k][l].w>Type.maxpw ||
							   W[i][j][k][l].w<Type.maxnw) // if so, reset!
							      W[i][j][k][l].w-=W[i][j][k][l].delta_w;

							W[i][j][k][l].delta_w = 0.0;   // Zero out the delta_w
						}
					}
				}
			}
		}

		// Here we can now set layer i's output and backprop again
		/*cout << "\n Back To layer: " << i << " => ";
		for(j=0;j<MAX_PE;++j)
			cout << sum[j] << ","; */

		Layer[i].SetOutput(sum);
		Layer[i].DualBackProp();

	}

	// If we just updated our weights, (that is the counter==epoch) then
	// we reset the counter back to 0
	if(count==Type.epoch)
		count=0;
} // End DualBackProp

// DualPartial
void NN::Dual2BackProp(double deriv[MAX_PE][MAX_PE])
// This function trains the Dual NN using a desired derivative
// matrix since we can have multiple inputs/outputs. The
// matrix follows a format similar to the PartialMatrix
// where if the NN maps X-->Y then the matrix is
// the derivative where:
// deriv[i][j] = desired_dYi/dXj typically.
// The training process is that for each output, we find
// a derivative vector, calculate the error vector:
// desired deriv - NN_dual's derv. and train the dual
// with a weight update rule. Since we may have to perform
// many passes, we might update each pass or create a
// adverage of each weight_delta and update at the end.
{
	double test[MAX_PE], sum[MAX_PE], err[MAX_PE];

	int i,j,k,l,m;

	for(m=0;m<NumOut;++m)	// m-for each output component
	{

		for(i=0;i<NumOut;++i)	// Form our component vector to backprop
		{
         // Added, transformed unit vector. We do this since we scale
         // the output. Therefore, we must compute the derivative of
         // the output function which is simply the output scale factor.

			test[i]=(i==m)?Type.OutScale[i]:0.0;
		}

		Layer[NumLayers-1].SetPartialOutput(test);  // Set output to component vector
		Layer[NumLayers-1].DualPartial();	 // Do a backprop at output layer

		for(i=NumLayers-2;i>=0;--i)	// i = to layer
		{
			for(j=0;j<MAX_PE;++j)	// Zero out sum vector
				sum[j]=0.0;

			for(j=(Type.feedforward)?NumLayers-1:i+1;j>i;--j) // j = from layer
				for(k=0;k<Layer[i].NumPE;++k)	// k - to PE
				{
					for(l=0;l<Layer[j].NumPE;++l)	// l - from PE
					{
						if(W[i][j][k][l].active)
							sum[k]+=Layer[j].BackPartial[l]*W[i][j][k][l].w;
					}
				}
			// Here we can now set layer i's output and backprop again
			Layer[i].SetPartialOutput(sum);
			Layer[i].DualPartial();
		}

		// If we're here then input has a backpropagated vector on it
		// we need to grab and place in the PartialMatrix in the
		// form: PartialMatrix[output_component][input_component]
		// This will be used for determining the matrix derivatives
		// in the form dA[i]dB[j] = dAdB[i][j]. Lastly, we need to
		// multiply it with the derivative of the scaling function
		// which is simple 1/InScale[i]... and remember that the we
		// need to work around the bias term at i=0.

		for(i=1;i<NumIn;++i)
			PartialMatrix[m][i-1] = Layer[0].BackPartial[i]/Type.InScale[i-1];

		// Next, we compare the vector of the partial matrix we just
		// filled with the same vector in the desired matrix and form
		// and error vector for the Dual that can be backprop'd.

		// Construct the error vector
		err[0]=0.0; // <== no bias error
		for(i=1;i<NumIn;++i)
			err[i] = deriv[m][i-1] - PartialMatrix[m][i-1];

		// Now we backprop this through a 2nd order backprop method
		DualTrain(err);
	}

} // End Dual2BackProp

// DualTrain
void NN::DualTrain(double e[])
// Similar to the DualBackProp function, this function trains
// the Dual NN to achieve a desired NN derivative.
{
	int i,j,k,l;

	double sum[MAX_PE];

	// First, we increment our counter... If the counter equals the epoch
	// size then the weights are updated (w=w+delta_w)
	dual_count++;

	// Next we need to convert our deriv_error back into unit-less values
	// by performing the inverse scaling:

	for(j=1;j<NumIn;++j)
	{
		e[j]=e[j]*Type.InScale[j-1];
		// cout << "\n Error: " << e[j];
	}

	// Now we can backpropagate our new unitless error vector thru the
	// 2nd order dual (where the activation functions are f'')

	Layer[0].SetDualOutput(e);  // Set input layer to error vector
	Layer[0].Dual2BackProp();	 // Do a dual2backprop at input layer

	for(i=1;i<NumLayers;++i)	// i = to layer
	{
		for(j=0;j<MAX_PE;++j)	// Zero out sum vector
			sum[j]=0.0;

		for(j=(Type.bias||Type.feedforward)?0:i-1;j<i;++j) // j = from layer
		{
			for(k=0;k<Layer[i].NumPE;++k)	// k - to PE
			{
				for(l=0;l<Layer[j].NumPE;++l)	// l - from PE
				{
					if(W[j][i][l][k].active)
					{
						// Determine backpropagated error
						sum[k]+=Layer[j].BackProp2[l]*W[j][i][l][k].w;

						// Calculate the delta_w for each active connection
						// delta_w+=(Beta/epoch)*dual_error*dual_partial

						W[j][i][l][k].delta_w = Layer[i].LayerType.Beta
                          /(double(Type.dual_epoch))*Layer[i].BackPartial[k]
                          *Layer[j].BackProp2[l];

						//cout << "\n BP:"<<i<<k<<"="<<Layer[i].BackPartial[k];
						//cout << "\n BP2:"<<j<<l<<"="<<Layer[j].BackProp2[l];
						//cout << "\n beta=" <<Layer[i].LayerType.Beta
                         // /(double(Type.dual_epoch));
						//cout << "\ndw"<<j<<i<<l<<k<<"="<<W[j][i][l][k].delta_w;

						if(dual_count==Type.dual_epoch)	// Is it time to update w?
						{
							// Update weight with delta_w
							W[j][i][l][k].w+=W[j][i][l][k].delta_w;
							/* //Are we out of weight range???
							if(W[i][j][k][l].w>Type.maxpw ||
							   W[i][j][k][l].w<Type.maxnw) // if so, reset!
							      W[i][j][k][l].w-=W[i][j][k][l].delta_w; */

							W[j][i][l][k].delta_w = 0.0;   // Zero out the delta_w
						}
					}
				}
			}
		}

		// Here we can now set layer i's output and backprop again

		Layer[i].SetDualOutput(sum);
		Layer[i].Dual2BackProp();

	}

	// If we just updated our weights, (that is the counter==epoch) then
	// we reset the counter back to 0
	if(dual_count==Type.dual_epoch)
		dual_count=0;

} // End TrainDual

// WeightUpdate
void NN::WeightUpdate()
// This works -- although not as efficient -- allows the update
{
   int i,j,k,l;

   for(i=0;i<MAX_LAYER;++i)         // from layer
      for(j=0;j<MAX_LAYER;++j)      // to layer
         for(k=0;k<MAX_PE;++k)      // from PE
            for(l=0;l<MAX_PE;++l)   // to PE
            {
               W[i][j][k][l].w+=W[i][j][k][l].delta_w;
               W[i][j][k][l].delta_w = 0.0;
            }
   // done updating..
} // End of WeightUpdate

// END
