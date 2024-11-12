#include <EloquentTinyML.h>
#include "best_model.h"

#define NUMBER_OF_INPUTS 7  // 7 features: Region, Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen
#define NUMBER_OF_OUTPUTS 2  // Binary classification for Channel (0 or 1)
#define TENSOR_ARENA_SIZE 3*1024  // Adjust this size based on your model's needs

Eloquent::TinyML::TfLite<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> ml;

float fResult[NUMBER_OF_OUTPUTS] = {0};
float fRes = NULL;

// True labels for the input examples (for comparison)
int trueLabels[] = {1, 0, 1, 0};  // Example true labels for all four inputs

// Initialize the output array
void initfResult(float *fResult) {
    for (int i = 0; i < NUMBER_OF_OUTPUTS; i++) {
        fResult[i] = 0.0f;
    }
}

// Display the output result
void displayOutput(float *fResult) {
    for (int i = 0; i < NUMBER_OF_OUTPUTS; i++) {
        Serial.print(fResult[i]);
        Serial.print(" ");
    }
    Serial.println();
}

// Compare predictions to true labels
void comparePredictions(int trueLabel, float *fResult) {
    // Find the predicted class based on the highest probability
    int predictedLabel = (fResult[0] > fResult[1]) ? 0 : 1;  // Assuming binary classification
    Serial.print("True Label: ");
    Serial.print(trueLabel);
    Serial.print(", Predicted Label: ");
    Serial.println(predictedLabel);
}

void setup() {
    Serial.begin(115200);
    ml.begin(Model);  // Initialize the model
}

void loop() {
    // Sample Inputs based on your dataset (Region, Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen)
    float input1[] = {1.0f, 0.112940f, 0.130727f, 0.081464f, 0.003106f, 0.065427f, 0.027847f};  // Example 1
    float input2[] = {1.0f, 0.062899f, 0.132824f, 0.103097f, 0.028548f, 0.080590f, 0.043455f};  // Example 2
    float input3[] = {0.0f, 0.120000f, 0.130000f, 0.090000f, 0.020000f, 0.060000f, 0.030000f};  // Example 3
    float input4[] = {0.0f, 0.110000f, 0.150000f, 0.085000f, 0.021000f, 0.075000f, 0.028000f};  // Example 4

    // Run inference for input1
    initfResult(fResult);
    fRes = ml.predict(input1, fResult);
    Serial.print("Input 1 Prediction: ");
    displayOutput(fResult);
    comparePredictions(trueLabels[0], fResult);

    // Run inference for input2
    initfResult(fResult);
    fRes = ml.predict(input2, fResult);
    Serial.print("Input 2 Prediction: ");
    displayOutput(fResult);
    comparePredictions(trueLabels[1], fResult);

    // Run inference for input3
    initfResult(fResult);
    fRes = ml.predict(input3, fResult);
    Serial.print("Input 3 Prediction: ");
    displayOutput(fResult);
    comparePredictions(trueLabels[2], fResult);

    // Run inference for input4
    initfResult(fResult);
    fRes = ml.predict(input4, fResult);
    Serial.print("Input 4 Prediction: ");
    displayOutput(fResult);
    comparePredictions(trueLabels[3], fResult);

    delay(5000);  // Delay between predictions
}
