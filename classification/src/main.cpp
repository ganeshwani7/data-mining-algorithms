//
//  main.cpp
//  dataMining
//
//  Created by Ganesh B Wani on 10/21/14.
//  Copyright (c) 2014 Ganesh B Wani. All rights reserved.
//

#include <iostream>
#include <sstream>
#include <fstream>
#include <mach-o/dyld.h>
#include <vector>
#include <ctime>
#include <random>
#include <cassert>
#include <math.h>
#include <iterator>
#include <map>
#include <fstream>

#define PI (22/7)

using namespace std;

typedef struct wine{
    // Tells which type of class it belongs to
    int class_;
    
    // Training dataset or Test dataset
    string category;
    
    //Remaining attributes present in the entry
    float alcohol, malic_acid, ash, alcalinity ;
    float magnesium;
    float total_phenols, flavenoinds, non_flavenoids, pro, color_intensity, hue, diluted;
    float proline;
    
    bool operator()( struct wine w1, struct wine w2){
        return w1.class_ < w2.class_;
    }
    
    float getEntry( int i){
        switch( i ){
            case 0:
                return class_;
                
            case 1:
                return alcohol;
                
            case 2:
                return malic_acid;
                
            case 3:
                return ash;
                
            case 4:
                return alcalinity;
                
            case 5:
                return magnesium;
                
            case 6:
                return total_phenols;
                
            case 7:
                return flavenoinds;
                
            case 8:
                return non_flavenoids;
                
            case 9:
                return pro;
                
            case 10:
                return color_intensity;
                
            case 11:
                return hue;
                
            case 12:
                return diluted;
                
            case 13:
                return proline;
                
        }
        cout << "Out of total number of entries";
        return 0;
    }
}wine;


bool comp_alcohol( wine w1, wine w2){
    return w1.alcohol < w2.alcohol;
}


bool comp_malic_acid( wine w1, wine w2){
    return w1.malic_acid < w2.malic_acid;
}

bool comp_ash( wine w1, wine w2){
    return w1.ash < w2.ash;
}

bool comp_alcalinity( wine w1, wine w2){
    return w1.alcalinity < w2.alcalinity;
}

bool comp_magnesium( wine w1, wine w2){
    return w1.magnesium < w2.magnesium;
}

bool comp_total_phenols( wine w1, wine w2){
    return w1.total_phenols < w2.total_phenols;
}

bool comp_flavenoinds( wine w1, wine w2){
    return w1.flavenoinds < w2.flavenoinds;
}

bool comp_non_flavenoids( wine w1, wine w2){
    return w1.non_flavenoids < w2.non_flavenoids;
}

bool comp_pro( wine w1, wine w2){
    return w1.pro < w2.pro;
}

bool comp_color_intensity( wine w1, wine w2){
    return w1.color_intensity < w2.color_intensity;
}

bool comp_hue( wine w1, wine w2){
    return w1.hue < w2.hue;
}


bool comp_diluted( wine w1, wine w2){
    return w1.diluted < w2.diluted;
}

bool comp_proline( wine w1, wine w2){
    return w1.proline < w2.proline;
}

//using arrptr = comp_array;
using fptr = bool(*)(wine w1,wine);
fptr array[14];

void getFuntionPointers( fptr *comp_array){
    
//    bool (*comp_array[13])( wine w1, wine w2);
//    fptr comp_array[13];
    comp_array[1] = comp_alcohol;
    comp_array[2] = comp_malic_acid;
    comp_array[3] = comp_ash;
    comp_array[4] = comp_alcalinity;
    comp_array[5] = comp_magnesium;
    comp_array[6] = comp_total_phenols;
    comp_array[7] = comp_flavenoinds;
    comp_array[8] = comp_non_flavenoids;
    comp_array[9] = comp_pro;
    comp_array[10] = comp_color_intensity;
    comp_array[11] = comp_hue;
    comp_array[12] = comp_diluted;
    comp_array[13] = comp_proline;
    
//    return comp_array;
}

template<typename T>
T ** allocateArray(int rows, int columns){
    T **array;
    array = new T*[rows];
    
    for( int i = 0;i < columns; i++){
        array[ i ] = new T[columns];
    }
    
    return array;
}

template<typename T>
void freeArray( T **array, int rows, int columns){
    for( int i = 0; i< rows; i++){
        delete [] array[i];
    }
    delete array;
}

template<typename T>
wine createWine( T *array){
    wine w;
    
    w.class_ = array[0];
    
    // Training dataset or Test dataset
    
    //Remaining attributes present in the entry
    w.alcohol = array[1];
    w.malic_acid = array[2];
    w.ash = array[3],
    w.alcalinity = array[4];
    w.magnesium = array[5];
    w.total_phenols = array[6];
    w.flavenoinds = array[7];
    w.non_flavenoids = array[8];
    w.pro = array[9];
    w.color_intensity = array[10];
    w.hue = array[11];
    w.diluted = array[12];
    w.proline = array[13];

    return w;
}

void loadData( float **array, string filename, int &row, int &column){
    string input = "abc,def,ghi";
    
//    wine wine1[100], wine2[100], wine3[100];
    
    istringstream ss(input);
    string token;
    row = 0; column = 0;
//    int row =0, column = 0;
//    cout << "File name is :" << filename.c_str()<< endl;
    
    ifstream file( filename.c_str());
    
    if( file.good()){
//        cout << " File is present on the device\n";
    }
    else{
        cout << "File is not present on the device\n";
        exit(0);
    }
    
    if (file.is_open()){
//        cout << "file is open\n";
    }
    string linebuffer;
    int flag = 0, actual_column = 0;
    
    while( file && getline(file, linebuffer)){
        istringstream ss( linebuffer.c_str());
        
        while( getline( ss , token, ',')) {
            array[row][column++] = atof( token.c_str() );
//            std::cout << token << ' ';
            //	cout<<row<<column<<" ";
        }
        
        //once you get the total number of columns after first iteration
        //no need of it lateron
        if( flag == 0){
            actual_column = column;
            flag = 1;
        }
        
        column= 0;
        ++row;
//         cout << endl;
        //cout << linebuffer << endl << endl;
    }
    column = actual_column;
}

struct RNG {
    int operator() (int n) {
        return static_cast<int>(std::rand()/(static_cast<double>(RAND_MAX)+1) * n);
    }
};

void distributeWine( vector<wine> &actual_wine, vector<wine> &train_wine, vector<wine> &test_wine){
    
    size_t const total_entries = actual_wine.size(),
    train_entries = (2* total_entries)/3,
    test_entries = total_entries - train_entries;

    random_shuffle( actual_wine.begin(), actual_wine.end(), RNG());
    
    int i = 0;
    while( i < train_entries){
        train_wine.push_back( actual_wine[i]);
        i++;
    }
    assert( i ==  train_wine.size());
    
    while( i < (test_entries+ train_entries) ){
        test_wine.push_back( actual_wine[i]);
        i++;
    }
    assert( (i - train_wine.size()) ==  test_wine.size());
}

void checkDataset( vector<wine> *wine_classes, vector<wine>*train_classes, vector<wine>*test_classes, int no_of_classes){
    for( int i = 0; i< no_of_classes; i++){
        assert(wine_classes[i].size() == (train_classes[i].size() + test_classes[i].size()) );
    }
    cout << "Assertion for elements check succeeded" << endl;
}

vector<wine> merge( vector<wine>*test, int size){
    vector<wine> merged_data;
    int i = 0;
//    cout << "total classes are : "<< size<< endl;
    while( i < size){
//        cout << "Merging data for iteration : "<< i <<endl;
        merged_data.insert( merged_data.end(), test[i].begin(), test[i].end());
        i++;
    }
    return merged_data;
}

bool comparison( wine w1, wine w2){
    return w1.alcohol == w2.alcohol;
}

void print_wine( wine w){
    cout << w.class_ << " ";
    cout << w.alcohol << " ";
    cout << w.malic_acid << " ";
    cout << w.ash << " ";
    cout << w.alcalinity << " ";
    cout << w.magnesium << " ";
    cout << w.total_phenols << " ";
    cout << w.flavenoinds << " ";
    cout << w.non_flavenoids << " ";
    cout << w.pro << " ";
    cout << w.color_intensity << " ";
    cout << w.hue << " ";
    cout << w.diluted << " ";
    cout << w.proline<< " ";
    cout << endl;
}

wine calculateStDev( vector<wine> wine_dataset, wine mean_wine, int class_number, int &vector_counter){
    wine std_deviation = {0};
    
    int counter = 0;
    while(1){
        if( wine_dataset.empty() || ( wine_dataset[vector_counter].class_ != class_number) ){
            break;
        }
        wine temp_wine = wine_dataset[vector_counter];
        std_deviation.alcohol +=        pow(temp_wine.alcohol - mean_wine.alcohol,2);
        std_deviation.malic_acid +=     pow(temp_wine.malic_acid - mean_wine.malic_acid,2);
        std_deviation.ash +=            pow(temp_wine.ash - mean_wine.ash,2);
        std_deviation.alcalinity +=     pow(temp_wine.alcalinity - mean_wine.alcalinity,2);
        std_deviation.magnesium +=      pow(temp_wine.magnesium - mean_wine.magnesium,2);
        std_deviation.total_phenols +=  pow(temp_wine.total_phenols - mean_wine.total_phenols,2);
        std_deviation.flavenoinds +=    pow(temp_wine.flavenoinds - mean_wine.flavenoinds,2);
        std_deviation.non_flavenoids += pow(temp_wine.non_flavenoids - mean_wine.non_flavenoids,2);
        std_deviation.pro +=            pow(temp_wine.pro - mean_wine.pro,2);
        std_deviation.color_intensity += pow(temp_wine.color_intensity - mean_wine.color_intensity,2);
        std_deviation.hue +=            pow(temp_wine.hue - mean_wine.hue,2);
        std_deviation.diluted +=        pow(temp_wine.diluted - mean_wine.diluted,2);
        std_deviation.proline +=        pow(temp_wine.proline - mean_wine.proline,2);
        ++vector_counter;
        ++counter;
    }
//    cout << "After iterations counter : "<< counter<< endl;
//    cout << "Vector counter : "<< vector_counter<< endl;
//    cout << "Proline content is as follows: "<< std_deviation.proline<< endl;
    // Calculate the average
        std_deviation.alcohol = sqrt(std_deviation.alcohol/ counter);
        std_deviation.malic_acid = sqrt( std_deviation.malic_acid/ counter);
        std_deviation.ash = sqrt( std_deviation.ash/ counter);
        std_deviation.alcalinity = sqrt( std_deviation.alcalinity /counter);
        std_deviation.magnesium = sqrt(std_deviation.magnesium / counter);
        std_deviation.total_phenols = sqrt(std_deviation.total_phenols / counter);
        std_deviation.flavenoinds = sqrt(std_deviation.flavenoinds / counter);
        std_deviation.non_flavenoids = sqrt(std_deviation.non_flavenoids  /counter);
        std_deviation.pro = sqrt(std_deviation.pro  /counter);
        std_deviation.color_intensity = sqrt(std_deviation.color_intensity  /counter);
        std_deviation.hue = sqrt(std_deviation.hue  /counter);
        std_deviation.diluted = sqrt(std_deviation.diluted  /counter);
        std_deviation.proline = sqrt(std_deviation.proline  /counter);
    return std_deviation;
}

wine calculateMean( vector<wine> wine_dataset, int class_number, int &vector_counter ){
    wine w = {0};
    int counter = 0;
    while(1){
        if( wine_dataset.empty() || ( wine_dataset[vector_counter].class_ != class_number) ){
            break;
        }
        w.alcohol +=        wine_dataset[vector_counter].alcohol;
        w.malic_acid +=     wine_dataset[vector_counter].malic_acid;
        w.ash +=            wine_dataset[vector_counter].ash,
        w.alcalinity +=     wine_dataset[vector_counter].alcalinity;
        w.magnesium +=      wine_dataset[vector_counter].magnesium;
        w.total_phenols +=  wine_dataset[vector_counter].total_phenols;
        w.flavenoinds +=    wine_dataset[vector_counter].flavenoinds;
        w.non_flavenoids += wine_dataset[vector_counter].non_flavenoids;
        w.pro +=            wine_dataset[vector_counter].pro;
        w.color_intensity += wine_dataset[vector_counter].color_intensity;
        w.hue +=            wine_dataset[vector_counter].hue;
        w.diluted +=        wine_dataset[vector_counter].diluted;
        w.proline +=        wine_dataset[vector_counter].proline;
        ++vector_counter;
        ++counter;
    }
//    cout << "Ash addition is"<< w.ash<< endl;
//    cout << "After iterations counter : "<< counter<< endl;
//    cout << "Vector counter : "<< vector_counter<< endl;
//    
//    cout << "Wine dataset size : "<< wine_dataset.size()<<endl;
//    cout << "Classes are : current: "<<  wine_dataset[vector_counter].class_ << " And incoming : "<< class_number<<endl;
    // Calculate the average
            w.alcohol /= counter;
            w.malic_acid /= counter;
            w.ash /= counter;
            w.alcalinity /= counter;
            w.magnesium /= counter;
            w.total_phenols /= counter;
            w.flavenoinds /= counter;
            w.non_flavenoids /= counter;
            w.pro /= counter;
            w.color_intensity /= counter;
            w.hue /= counter;
            w.diluted /= counter;
            w.proline /= counter;
    return w;
}

void printDataset( vector<wine> dataset){
    for( int i =0; i< dataset.size(); i++){
        print_wine( dataset[i] );
    }
}

//For this function to work, vector should be in a sorted order

float getClassProbabilities( vector<wine>::const_iterator &it, int wine_class, size_t total_wine_length){
    int count = 0;
    float probability;
    wine w = *it;
    while( w.class_ == wine_class){
        ++it;
        ++count;
        w = *it;
    }
    
//    cout << "count is :"<< count << endl;
    
    if( count!= 0){
        probability = ( (float)count/ total_wine_length);
    }
//    cout << "Probability is : "<< probability << endl;
    assert(probability !=0);
    return probability;
}

// Actual function to calculate gaussian distribution from actual formula

float getProbaility( float x, float mean, float deviation){
    float exponent = exp( -( pow( x- mean, 2) / ( 2* pow(deviation, 2) ) ) );
    
    return ( ( 1/ ( sqrt( 2* PI) * deviation) ) * exponent);
}

// This function calculates P( X/Ci ) for each class and each attribute of the tuple
// Multiplies the probability obtained for attributes of specific classes

int getErrorCount( wine w, map<int, wine> mean_wine_map, map<int, wine>sdev_wine_map, map<int, float> prob_wine_map, int current_class){
    assert(current_class == w.class_);
    map<int, float> probability_map;
    float current_class_probabilty = 1; // 1 just because for further multiplication
    
    for( int i=1; i<= mean_wine_map.size(); i++){
        wine temp_mean_wine = mean_wine_map[i], temp_dev_wine = sdev_wine_map[i];
        current_class_probabilty *= getProbaility( w.alcohol, temp_mean_wine.alcohol, temp_dev_wine.alcohol);
        current_class_probabilty *= getProbaility( w.malic_acid, temp_mean_wine.malic_acid, temp_dev_wine.malic_acid);
        current_class_probabilty *= getProbaility( w.ash, temp_mean_wine.ash, temp_dev_wine.ash);
        current_class_probabilty *= getProbaility( w.alcalinity, temp_mean_wine.alcalinity, temp_dev_wine.alcalinity);
        current_class_probabilty *= getProbaility( w.magnesium, temp_mean_wine.magnesium, temp_dev_wine.magnesium);
        current_class_probabilty *= getProbaility( w.total_phenols, temp_mean_wine.total_phenols, temp_dev_wine.total_phenols);
        current_class_probabilty *= getProbaility( w.flavenoinds, temp_mean_wine.flavenoinds, temp_dev_wine.flavenoinds);
        current_class_probabilty *= getProbaility( w.non_flavenoids, temp_mean_wine.non_flavenoids, temp_dev_wine.non_flavenoids);
        current_class_probabilty *= getProbaility( w.pro, temp_mean_wine.pro, temp_dev_wine.pro);
        current_class_probabilty *= getProbaility( w.color_intensity, temp_mean_wine.color_intensity, temp_dev_wine.color_intensity);
        current_class_probabilty *= getProbaility( w.hue, temp_mean_wine.hue, temp_dev_wine.hue);
        current_class_probabilty *= getProbaility( w.diluted, temp_mean_wine.diluted, temp_dev_wine.diluted);
        current_class_probabilty *= getProbaility( w.proline, temp_mean_wine.proline, temp_dev_wine.proline);
        
        
        current_class_probabilty *= prob_wine_map[i];
        probability_map[i] = current_class_probabilty;
        current_class_probabilty = 1;
    }
//    cout << "All probabilities are : ";
    
    float max = 0.0;
    int class_of_wine = 1;
    for( auto it: probability_map){
//        cout << it.second<< " ";
        if( it.second > max){
            class_of_wine = it.first;
            max = it.second;
        }
    }
    
//    cout << "Class of wine from gaussian : "<< class_of_wine << " and actual class : "<< current_class<< endl;
    
    if( class_of_wine == current_class)
        return 0;
    else
        return 1;
}

float calculateNaiveBayesian( vector<wine>train_dataset, vector<wine> test_dataset, int wine_classes){
    map<int, wine> mean_wine_map, sdev_wine_map;
    map<int, float>prob_wine_map;
    
    cout << "Addition of training and test dataset is : "<< train_dataset.size() + test_dataset.size() << endl;
    
    for( int i= 1; i<= wine_classes; i++){
        mean_wine_map.insert( pair<int, wine>(i, wine()));
        sdev_wine_map.insert( pair<int, wine>(i, wine()));
    }
    
    
    int vector_counter = 0, error_count = 0;
    
    for( int i= 1; i<= wine_classes; i++){
        mean_wine_map[i] = calculateMean(train_dataset, i, vector_counter);
    }
    
//    cout << "Means for each attribute is as follows:\n\n";
//    for( int i= 1; i<= wine_classes; i++){
//        print_wine( mean_wine_map[i]);
//    }
    
    vector_counter = 0;
    for( int i= 1; i<= wine_classes; i++){
        sdev_wine_map[i] = calculateStDev(train_dataset, mean_wine_map[i], i, vector_counter);
    }

//    cout << "\nStandard Deviation for each attribute is as follows:\n\n";
//    for( int i= 1; i<= wine_classes; i++){
//        print_wine( sdev_wine_map[i]);
//    }
    
    vector<wine>::const_iterator it = train_dataset.cbegin();
    
    
//    printDataset( train_dataset);
    
    for( int i= 1; i<= wine_classes; i++){
        prob_wine_map[i] = getClassProbabilities( it, i, train_dataset.size());
    }

    
    float prob_addition = 0;
    for( auto it  : prob_wine_map){
//        cout << "it.second is : "<< it.second<< endl;
        prob_addition += it.second;
    }
    
//    cout << "Probability addition is : " << prob_addition <<  endl;
    
    assert( ( prob_addition) == 1);
    
    for( auto test_iter : test_dataset ){
        error_count += getErrorCount( test_iter, mean_wine_map, sdev_wine_map, prob_wine_map, test_iter.class_);
    }
    cout << "Error count "<< error_count << endl;
    cout<< "error rate is "<< ( (float)error_count/(float)test_dataset.size() ) << endl;
    
    return  ( (float)error_count/(float)test_dataset.size() ) ;
}

void calculateKFoldNaive( vector<wine>wine_classes[3], int total_wine_classes, const int k){
    int temp = 0;
    float total_error_rate = 0;
    vector<wine> all_wines;
    vector<wine> train_dataset, test_dataset;
    
    cout << "About to distribute wine\n";
    
    for( int i=0; i< total_wine_classes; i++){
        for( wine w: wine_classes[i]){
            all_wines.push_back( w);
//        distributeWine( wine_classes[i], train_classes[i], test_classes[i]);
        }
    }
    random_shuffle(all_wines.begin(), all_wines.end(), RNG());
    
    cout << "After random shuffling\n";
    
    float chunk_size = (int)all_wines.size()/k;
    cout << "Chunk size is : " << chunk_size << endl;
    
    int count = 0;
    while( temp < k ){
        for( wine w: all_wines){
            if( count >= ( temp * chunk_size) && count < ( (temp+1) *chunk_size )){
                test_dataset.push_back(w);
            }
            else
                train_dataset.push_back(w);
            ++count;
        }
        wine w;
        sort( train_dataset.begin(), train_dataset.end(), w);
        total_error_rate += calculateNaiveBayesian(train_dataset, test_dataset, total_wine_classes);
        count = 0;
        temp++;
        train_dataset.clear();
        test_dataset.clear();
    }
    cout << "Average error rate by 10-fold Naive Bayesian is : " << total_error_rate/k << endl;
}


void addInDiscrete( vector<wine>train_dataset, vector<wine> &discretized_dataset, int i){
    auto it = train_dataset.begin();
    int counter = 0;
    while( it != train_dataset.end()){
//        cout << " Classnumbers are : "<< it->class_ << "and "<< (it+1)->class_<<endl;
        // Class and next entry should not be the same
        if( it->class_ != (it+1)->class_  && it->getEntry(i) != (it+1)->getEntry(i) ){
            if( !discretized_dataset.empty() && discretized_dataset.back().class_ !=  (it+1)->class_ &&
               discretized_dataset.back().getEntry(i) != it->getEntry(i)){
          //      cout << "Inside if statement\n";
                discretized_dataset.push_back( *it);
                ++counter;
            }else if( discretized_dataset.empty()){
            //    cout << "Inside else statement\n";
                discretized_dataset.push_back( *it);
                ++counter;
            }
        }
        ++it;
        if( counter == 4){
            break;
        }
    }
}

void makeThemDiscrete( vector<wine> &train_dataset, vector<wine> &discretized_dataset, int total_classes, int total_attributes){
    fptr comp_array[14];
    getFuntionPointers( comp_array);
    
    for( int i = 1; i <= total_attributes; i++){
        sort( train_dataset.begin(), train_dataset.end(), comp_array[i]);
//        cout << "Sorting iteration : "<< i <<endl;
//        printDataset( train_dataset);
        addInDiscrete( train_dataset, discretized_dataset, i);
    }
    
    cout << "\nDiscretized dataset is : \n";
    printDataset(discretized_dataset);
    cout << "Train dataset size: " << train_dataset.size() << endl;
    cout << "Discretized dataset size: " << discretized_dataset.size() << endl;
    
}

//template<typename T>
void writeVectorInFile( vector<wine> train_dataset, string filename, int total_attributes){
    ofstream myFile;
    
//    filename += ".txt";
    myFile.open( filename );
//    cout << "\nTotal attributes are: "<<  total_attributes<<"\n";
    
    for( auto w : train_dataset){
        for( int i = 0; i <= total_attributes; i++){
            myFile << w.getEntry(i);
            if( i != total_attributes )
                myFile << ",";
        }
        myFile << "\n";
    }
    myFile.close();
}

int main(int argc, const char * argv[]) {
    // insert code here...
//    std::cout << "Hello, World!\n";
    
    // Initialise the seed for random number
    srand(unsigned(time(NULL)));
    
    int rows = 1000, columns = 1000;
    string filename;

    if(argc != 2){
        cout << " Format : ./main <filename>\n";
        exit(1);
    }
    
    float **array = allocateArray<float>(rows, columns);

    int total_wine_classes = 3, total_attributes = 13;
    
    // Contains all the wine entries according to their classes
    vector<wine> wine_classes[3 ];
    
    // Contains discretized values for the training wine dataset
    vector<wine> discretized_wine;
    
    // contains array of vectors for the training entries and test entries
    vector<wine>train_classes[3 ], test_classes[3 ];
    
    loadData( array, argv[1], rows, columns);

    int current_index = 0;
    for( int row = 0; row < rows; row++){
        // Because first column always indicates the class current entry belongs to
        current_index = array[row][0];
        wine_classes[ current_index -1 ].push_back(createWine( array[row]));
    }
    
    // Radomly partition the entries in training and testing dataset
    for( int i=0; i< total_wine_classes; i++){
        distributeWine( wine_classes[i], train_classes[i], test_classes[i]);
    }
    
    checkDataset( wine_classes, train_classes, test_classes, total_wine_classes);
    
//    cout<<" after check dataset";
    vector<wine>train_dataset = merge( train_classes, total_wine_classes);
    vector<wine>test_dataset = merge( test_classes, total_wine_classes);
    
    writeVectorInFile( test_dataset, "dataset/wine.test", total_attributes);
    writeVectorInFile( train_dataset , "dataset/wine.data", total_attributes);
    
    cout << "Train dataset size : " << train_dataset.size()<<endl;
    cout << "Test dataset size : " << test_dataset.size()<<endl;

    calculateNaiveBayesian( train_dataset, test_dataset, total_wine_classes);

    calculateKFoldNaive( wine_classes, total_wine_classes, 10);
    
    wine w;
    sort( train_dataset.begin(), train_dataset.end(), w);
    sort( test_dataset.begin(), test_dataset.end(),  w);
    
    
    fptr comp_array[14];
    getFuntionPointers(comp_array);
    
    random_shuffle( train_dataset.begin(), train_dataset.end(), RNG());
    
    makeThemDiscrete( train_dataset, discretized_wine, total_wine_classes, total_attributes);
        cout << "After make them discrete\n";
    
    freeArray( array, rows, columns);
 
    cout << "Number of rows and columns are as follows\n";
    cout << rows << " " << columns<< endl;
    
//    documentDataSet( array, rows, columns);
    
    return 0;
}
