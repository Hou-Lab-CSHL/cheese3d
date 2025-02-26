/*
  Open Valve

*/
#include <Wire.h>

const int buttonPin =  52; // Button to trigger recording
bool record = false;
uint8_t buttonState;

void setup() {
  pinMode(buttonPin, INPUT);
  Serial.begin(9600);
}

void loop() {
  buttonState = digitalRead(buttonPin);
  if (buttonState) {
    Serial.println("trigger");
    // toggle recording state on or off
    // record = !record;
  }
  // if (record) {
  //   // send pulse to keep bonsai sampling
  //   Serial.println("pulse");
  // }
  // delay(5); // half 100 fps period
}
