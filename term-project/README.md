## Group members
* Josiah Campbell
* Timothy Eggleston
* Vincent Ball 

## Running the Project
The data is big! The following script will unzip a CSV file and run the classifier
```sh
./unzip_and_run
```

* If you get an error, try `chmod u+x classifier` and re-rerun. 

## Getting Started
To install new dependencies
```
pip3 install <library>
```

To retrieve existing requirements
```
pip3 install -r requirements.txt
```

To add new pip requirements
```
pip3 freeze > requirements.txt
```

## Converting ActiveRecord to CSV

Run `wc -l <filename.csv>` to confirm loc in a file

Run `rails runner <script.rb>` to extract ActiveRecord messages
