#include<iostream>
#include<vector>
#include<string>
#include<unordered_set>
#include<fstream>
#include<sstream>
using namespace std;

string dynamicfile = "./pretrined/dynamics.names";
string classfile = "./pretrained/mscoco_labels.names";
vector<string> classes;
unordered_set<string> dynamicClasses_set;
vector<string> dynamicClasses_vec;

int main(){

    cout<<"dynamic txt file"<<endl;
    ifstream ifs2(dynamicfile.c_str());
    string line;
    while(getline(ifs2,line)){  
        dynamicClasses_set.insert(line);
        dynamicClasses_vec.push_back(line);
    }

    cout<< "start" <<endl;
    for(int i = 0; i<dynamicClasses_vec.size(); i++){
        string name = dynamicClasses_vec[i];
        cout << name << " isTrue: "<<dynamicClasses_set.count(name) <<endl;
    }
    cout<<"end"<<endl;

    cout<<"classes file"<<endl;
    ifstream ifs(classfile.c_str());
    string line;
    while (getline(ifs, line)){
        classes.push_back(line);
        cout<<line<<endl;
    }

    return 0;
}