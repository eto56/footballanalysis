
cpp_file = "model.cpp"

include_dir = "/usr/include/"

openGL = $(include_dir)GL/

nlohmann_json = $(include_dir)nlohmann/


model: clean
	g++ -o model $(cpp_file) -I$(openGL) -lGL -lGLU -lglut -lm -I$(include_dir)  -I$(nlohmann_json) 


clean:
	if [ -f model ]; then rm model; fi
	if [ -f *.o ]; then rm *.o; fi
	