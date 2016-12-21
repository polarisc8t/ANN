/* 
 * Simple NN application
 * reads a training file and cfg file and test file
 * steps
 * 1. loads cfg file, creates internal NN
 *    (cfg file can point to a weights file for preloading weights)
 * 2. loads training file
 * 3. begins training the NN based on the training file
 *   training stops after specified interations or fitness is met (as defined in cfg file)
 * 4. after training, the overall fitness report outputted.
 * 5. The test file is loaded. it contains various input vectors to get NN outputs.
 * 6. The NN produces an output for every input specified in the test file
 * 7. The test inputs and computed outputs are outputted.


FORMAT: nn.exe <cfg> <training_file> <test_file> <name>
OUTPUT: 
    <name>.training.out
    <name>.test.out

*/

#include "nn.h"

#define MAX_TRAIN_ENTRIES 42000
#define MAX_TEST_ENTRIES 42000

int LoadTestFile(char*);
int LoadTrainFile(char*);
double Fitness();
int DumpResults(char*);

// Global variables
NN mynn;
int numin, numout;
long unsigned int cycles, max_cycles=1001;
double err[MAX_PE], error, fitness, 
       target_fitness=0.95;

// training and test tables
double  train_table_in[MAX_TRAIN_ENTRIES][MAX_PE],    // training table contains a list of input/output pairs to train with
        train_table_out[MAX_TRAIN_ENTRIES][MAX_PE],
        test_table_in[MAX_TEST_ENTRIES][MAX_PEx2],      // test table contains a list of input/output pairs to test with
        test_table_out[MAX_TEST_ENTRIES][MAX_PEx2];
int train_entries, test_entries,
    training_epoch=1000; // evaluate fitness every this many cycles

int main(int argc, char *argv[]) {
    char *cfgfile, *trainfile, *testfile, *nametag;
    int i;

    if(argc != 7) {
        cout << "\n-E- Format: nn.exe <max_cycles> <target_fitness: max 1.0> <cfg_file> <train_file> <test_file> <name_tag>\n";
        exit(1);
    }

    max_cycles = atoi(argv[1]);
    target_fitness = atof(argv[2]);
    cfgfile = argv[3];
    trainfile = argv[4];
    testfile = argv[5];
    nametag = argv[6];

    cout << "\ncfg=" << cfgfile
         << "\ntrain=" << trainfile
         << "\ntest=" << testfile
         << "\nname=" << nametag;


    // Step 1: Load the NN cfg
    cout << "\n-I- Loading NN cfg";
    mynn.InitNN(cfgfile);

    // Step 2: Load Training file and test file
    LoadTrainFile(trainfile);
    LoadTestFile(testfile);
    // We should check that the training file is compatible with the NN (inputs and outputs line up!)

    // Step 3: Begin training until criteria is met
    int r; // random pointer to training table
    cout << "\n-I- Running for " << max_cycles;
    fitness = 0.0; // make it large!
    for(cycles = 0; cycles < max_cycles && fitness <= target_fitness ; cycles++) {
        r = int(rand()%train_entries);
        mynn.SetInput(train_table_in[r]);
        mynn.Evaluate();
        error = 0.0;
        for(i=0;i<=numout;i++) {
            err[i] = -mynn.Output[i]+train_table_out[r][i];
        }
        mynn.DualBackProp(err); // train with error vector
        // get fitness every so many cycles
        
        /* cout << "\n-I- " << cycles << ": Input={";
        for(int j=0;j<numin;j++) {
            cout << " " << train_table_in[r][j];
        }
        cout << "}, Target=" << train_table_out[r][0] << ", NN=" << mynn.Output[0] << ", Error = " << err[0];
	*/        

        if(cycles%training_epoch == 0) {
            fitness = Fitness();
            cout << "\n-I- " << cycles << ": Fitness = " << fitness;
        }
    }

    // Step 4: load the test file and dump the results of the test
    DumpResults(nametag);

    cout << "\n-I- Done.\n";
    return 0;
}

double Fitness() {
    int i,j;
    double err[MAX_PE], error, fitness;

    // try: fitness = 1/(err+1)
    // where err is SUM(ABS(err[i][j]))/(i*j) where i is over all entries and j is over all outputs

    // using test table
    error = 0.0;
    for(i=0; i < test_entries; i++) {
        mynn.SetInput(test_table_in[i]);
        mynn.Evaluate();
        for(j=0;j<=numout;j++) {
            err[j] = mynn.Output[j]-test_table_out[i][j];
            if(err[j] < 0.0) {
                error += -err[j];
            } else {
                error += err[j];
            }
        }
    }
    error = error/(test_entries*numout);
    fitness = 1.0/(1.0+error);
    return fitness;
}

int LoadTestFile(char *fn)
{
    int i,j, entries, inputs, outputs;
    double v;
    FILE *inf = fopen(fn, "r");

    cout << "\n-I- LoadTestFile: Reading table file: " << fn;

    // read entries, inputs, outputs
    fscanf(inf,"Entries: %d\n", &entries);
    fscanf(inf,"Inputs: %d\n", &inputs);
    fscanf(inf,"Outputs: %d\n", &outputs);

    cout << "\n-D- Test Entries="<<entries<<", Inputs="<<inputs<<", Outputs="<<outputs;

    test_entries = entries;

    for(i=0; i < entries ; i++) {
        // cout << "\n-D- Inputs={";
        for(j=0; j < inputs ; j++) {
            fscanf(inf, "%lf",&v);
            test_table_in[i][j] = v;
            // cout << " " << v;
        }
        // cout << "}, Outputs={";
        for(j=0; j < outputs ; j++) {
            fscanf(inf, "%lf",&v);
            test_table_out[i][j] = v;
            // cout << " " << v;
        }
        // cout << "}";
    }

    fclose(inf);
    return 1;
}

int LoadTrainFile(char *fn) {
    int i,j, entries, inputs, outputs;
    double v;
    FILE *inf = fopen(fn, "r");

    cout << "\n-I- LoadTrainFile: Reading table file: " << fn;

    // read entries, inputs, outputs
    fscanf(inf,"Entries: %d\n", &entries);
    fscanf(inf,"Inputs: %d\n", &inputs);
    fscanf(inf,"Outputs: %d\n", &outputs);

    cout << "\n-D- Train Entries="<<entries<<", Inputs="<<inputs<<", Outputs="<<outputs;

    train_entries = entries;
    numin = inputs;
    numout = outputs;

    for(i=0; i < entries ; i++) {
        // cout << "\n-D- Inputs={";
        for(j=0; j < inputs ; j++) {
            fscanf(inf, "%lf",&v);
            train_table_in[i][j] = v;
            // cout << " " << v;
        }
        // cout << "}, Outputs={";
        for(j=0; j < outputs ; j++) {
            fscanf(inf, "%lf",&v);
            train_table_out[i][j] = v;
            // cout << " " << v;
        }
        // cout << "}";
    }

    fclose(inf);
    return 1;
}

int DumpResults(char *fn) {

    int i,j;
    double err[MAX_PE], error;
    cout << "\n-I- Dumping Results";

    // using test table
    for(i=0; i < test_entries; i++) {
        mynn.SetInput(test_table_in[i]);
        mynn.Evaluate();
        error = 0.0;
        for(j=0;j<=numout;j++) {
            err[j] = mynn.Output[j]-test_table_out[i][j];
            error += err[j];
        }
        error = error/numout;
        // get average error: avgerr = sum of all error components / numout
        // if avgerr meets criteria, then exit loop (cycles = max_cycles)
        // else, train NN: mynn.DualBackProp(error[])
        cout << "\n-Test["<<i<<"]: {";
	if(numin < 10) {	// for large inputs, don't bother showing... 
        	for(j=0;j<numin;j++) {
            		cout << " " << test_table_in[i][j];
        	}
	} 
        cout << "}, Target="<<test_table_out[i][0] << ", NN=" << mynn.Output[0] << ": Error = " << error;
    } 
    // Save weights
    mynn.SaveWeights("weights.wts");

    return 1;
}
