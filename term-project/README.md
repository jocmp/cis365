## Group members
* Josiah Campbell
* Timothy Eggleston
* Vincent Ball 

## Running the Project
The following script will attempt to download pip3 requirements and run the classifier.
```sh
./run_all.sh
```

* If you get an error, try `chmod u+x run_all.sh` and re-rerun.

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
