void setup() {
  pinMode(7, OUTPUT);
}
void loop() {
  randomSeed(analogRead(5));
  int random_time_gap = random(1000);
  digitalWrite(7, HIGH);  
  delay(10);              
  digitalWrite(7, LOW);  
  delay(9500+random_time_gap);            
}

