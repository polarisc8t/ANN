/************************************************
 NN PROJECT BY AKZCHEUNG
 LAYER DEFINITION
*************************************************/

#define MAX_LAYER 5

#include <iostream>
#include <math.h>

using namespace std;

// ACTIVATION FXNS

double utanh(double x)		// User-defined tanh(x) function
{
	double y;
	if(x > 9.2) y= 1.0;
	else if (x < -9.2) y = -1.0;
	else y = tanh(x);

	return y;
}

double udtanh(double x)		// user-defined tanh'(x)
{
	double y = 4.0*exp(2.0*x)/
		       ((exp(2.0*x)+1)*(exp(2.0*x)+1));

	return y;
}

double d2tanh(double x)    // tanh''(x)
{
	double y=exp(2.0*x),z;
	z = 8.0*y*(1.0-y)/((y+1)*(y+1)*(y+1));
	return z;
}

double sigmoid(double x)	// sigmoid function, S(x)
{
	return 1.0/(1.0+exp(-x));
}

double dsigmoid(double x)	// S'(x)=S(x)(1-S(x))
{
	// It is probably quicker computationally to use
	// the formula: S'(x) = -exp(-x)/(1+exp(-x))^2  (?)
	double y = sigmoid(x);

	return y*(1-y);
}

double d2sigmoid(double x)  // S''(x) = S'(x)(1-2*S(x))
{
	double y = sigmoid(x), z=dsigmoid(x);

	return z*(1.0-2.0*y);
}

// END OF ACTIVATION FNs.


// NNLayerType

struct NNLayerType
{
	int NumPE,		// Num of PE in layer
		ActType,    // PE activation types
					// 0-Linear, 1-TanH, 2-Sigmoid, 3-Gaussian(?)
		OutType;	// 0-Linear, 1-One Hot, 2-Softmax

	double	 DerivOff,     // Derivative Offset for training
			 	 Beta,         // Learning coefficient
				 Momentum;     // Learning momentum

	// Activation function parameters could go here
};


// NNLayer CLASS

class NNLayer
{
	public:
		NNLayer();				 // Default constructor
		void InitNNLayer(NNLayerType);   // Inits the NNLayer
		void Evaluate();            // Determines layer state and output
		void SetInput(double[]);    // Sets the layer input
		void SetOutput(double[]);   // Sets the layer output for backprop
		void SetDualOutput(double[]);   // Sets the layer dualoutput for dual train
		void SetPartialOutput(double[]);   // Sets the partial backprop output
		void DualPartial();     // Sets backprop'd vector for partial Derive extraction
		void DualBackProp();     // Sets backprop'd vector for backpropagation training
		void Dual2BackProp();    // for training the dual

		// These member data are accessible publicly

		double Fprime[MAX_PE],     // derivative of each PE
			   F2prime[MAX_PE],	   // second derivative of each PE
				 Output[MAX_PE],     // output vector
				 DualOutput[MAX_PE],  // output of the Dual vector
				 PartialOutput[MAX_PE],     // partial output vector
				 Input[MAX_PE],      // input vector
				 BackProp[MAX_PE],   // backprop vector
				 BackProp2[MAX_PE],  // backprop vector for the dual
				 BackPartial[MAX_PE]; // backprop vect for Partials.


		int NumPE;						// copied here for ease

		NNLayerType LayerType;     // NNLayer parameters
};

// ********************************************
// NNLayer CLASS Definition of member functions

// Constructor
NNLayer::NNLayer()
{
	// Here we zero out our vectors
	int i;

	for(i=0;i<MAX_PE;++i)
	{
		Fprime[i] = 0.0;
		F2prime[i] = 0.0;
		Output[i] = 0.0;
		DualOutput[i] = 0.0;
		Input[i] = 0.0;
		BackProp[i] = 0.0;
		BackProp2[i] = 0.0;
		BackPartial[i] = 0.0;
	}
}  // End Constructor

// InitNNLayer
void NNLayer::InitNNLayer(NNLayerType Type)
{
	NumPE=Type.NumPE;
	LayerType=Type;
	// cout << "\nLayer: "<<NumPE<<"\n";
	// Inits the NumPE and makes a copy of the LayerType packet
}

// Evaluate
void NNLayer::Evaluate()
{

	int i, max=0;

	double net=0.0;	// Net of PEs' nets for Softmax outputs.

	for(i=0;i<NumPE;++i)	// Evaluate each PE in layer
	{

		// Here we evaulate the output of each PE along with F'
		switch(LayerType.ActType)
		{
			case 0 : // Linear Output : f(net) = net
						Output[i] = Input[i];
						Fprime[i] = 1.0;
						F2prime[i] = 0.0;
						break;
			case 1 : // Hyperbolic Tangent
						// Form: f(net) = tanh(net)
						Output[i] = utanh(Input[i]);
						Fprime[i] = udtanh(Input[i]);
						F2prime[i] = d2tanh(Input[i]);
						break;
			case 2 : // Sigmoid
						// Form: 1/(1+EXP(net))
						Output[i] = sigmoid(Input[i]);
						Fprime[i] = dsigmoid(Input[i]);
						F2prime[i] = d2sigmoid(Input[i]);
						break;
			case 3: // Sinusoidal (test)
						Output[i] = sin(Input[i]);
						Fprime[i] = cos(Input[i]);
						F2prime[i] = -sin(Input[i]);
						break;
			case 4: // Gaussian (test)
						Output[i] = exp(-(Input[i]*Input[i]));
						Fprime[i] = -2.0*Input[i]*Output[i];
						F2prime[i] = (4.0*Input[i]*Input[i]-2.0)*Output[i];
						break;
			default : cout << "\n Activation Type Not Supported...\n";
		}

		if(LayerType.OutType==2)	// Softmax output activated?
			net+=Output[i];			// Get output sum

		else if(LayerType.OutType==1) // Determine One-Hot output
			if(Output[i]>Output[max])
				max=i;

	}

	if(LayerType.OutType==2)	// Softmax normalization: exp(a-sum(a))
		for(i=0;i<NumPE;++i)
			Output[i] = exp(Output[i]-net);

	else if(LayerType.OutType==1)	// One-Hot Output
		for(i=0;i<NumPE;++i) {
			if(i!=max) {
				Output[i]=0.0;		// Only select the highest
			} else {
				Output[i]=1.0;		// output to be a 1, the rest 0's.
			}
		}

} // END Output

// SetInput
void NNLayer::SetInput(double inp[])
// Sets the Input vector.. this can also be done publicly
{
	int i;

	for(i=0;i<NumPE;++i)
		Input[i] = inp[i];

} // END SetInput

// SetOutput
void NNLayer::SetOutput(double outp[])
// Sets the Output vector.. this can also be done publicly
{
	int i;

	for(i=0;i<NumPE;++i)
		Output[i] = outp[i];

} // END SetOutput

// SetDualOutput
void NNLayer::SetDualOutput(double outp[])
// Sets the DualOutput vector.. this can also be done publicly
{
	int i;

	// cout << "\n SetDualOutput:";
	for(i=0;i<NumPE;++i)
	{
		DualOutput[i] = outp[i];
		// cout << " " << outp[i];
	}

} // END SetDualOutput

// SetPartialOutput
void NNLayer::SetPartialOutput(double outp[])
// Sets the Output vector.. this can also be done publicly
{
	int i;

	for(i=0;i<NumPE;++i)
		PartialOutput[i] = outp[i];

} // END SetPartialOutput

// DualPartial layer
void NNLayer::DualPartial()
{
	int i;

	for(i=0;i<NumPE;++i)
		BackPartial[i] = PartialOutput[i]*Fprime[i];

} // End DualPartial()

// DualBackProp layer
void NNLayer::DualBackProp()
{
	int i;

	for(i=0;i<NumPE;++i)
		BackProp[i] = Output[i]*(Fprime[i]+LayerType.DerivOff);

} // End DualBackProp()

// DualBackProp layer
void NNLayer::Dual2BackProp()

{
	int i;
//	cout << "\n BackProp2: ";
	for(i=0;i<NumPE;++i)
	{
		BackProp2[i] = DualOutput[i]*Fprime[i];
//		cout << " " << BackProp2[i];
	}

} // End Dual2BackProp()

// END
