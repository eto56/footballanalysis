#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <thread>
#include <chrono>
#include <cmath>

#include <GL/glut.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct Detection {
    float x;
    float y;
    int r;
    int g;
    int b;
    int isplayer;
};

using FrameDetections = std::vector<Detection>;
std::map<int, FrameDetections> all_frames;
std::vector<int> sorted_frames;
int current_frame_index = 0;

constexpr float SCALE_FACTOR = 0.01f;

 
int last_mouse_x = -1;
int last_mouse_y = -1;
bool left_button_down = false;

float field_width = 105.0f;
float field_height = 68.0f;

void rescaleDetection(Detection &det) {
    // 0-1の範囲を-0.5-0.5に変換してから、フィールドのサイズにスケーリング
     //std::cout << "x: " << det.x << " y: " << det.y << std::endl;

    det.x -= 0.5f;
    det.y -= 0.5f;
 
    det.x *= field_width;
    det.y *= field_height;

    det.x *= SCALE_FACTOR;
    det.y *= SCALE_FACTOR;

   // std::cout << "x: " << det.x << " y: " << det.y << std::endl;
    
}
void drawSphereAtDetection(const Detection& det) {
    glColor3ub(det.r, det.g, det.b);
    glPushMatrix();
    float sx = det.x ;
    float sy = det.y ;
    glTranslatef(sx, sy, 0.005f);
    glutSolidSphere(0.005, 16, 16);
    glPopMatrix();
}

void drawHumanModel(const Detection& det) {

    if (det.isplayer==0 ){
        // draw ball
        glColor3ub(det.r, det.g, det.b);
        glPushMatrix();
        float sx = det.x ;
        float sy = det.y ;
        float rad = 0.005f;
        glTranslatef(sx, sy, rad);
        glutSolidSphere(rad, 16, 16);
        glPopMatrix();

        std::cout << "draw ball" << std::endl;
       
        return;
    }

    double human_scale = 0.1;
     glColor3ub(det.r, det.g, det.b);

    glPushMatrix();
     
    glTranslatef(det.x, det.y, 0.0f);
 
    glScalef( human_scale, human_scale, human_scale);

    
    glLineWidth(2.0f);
    glBegin(GL_LINES);

      // 足(足元) -> 膝
      glVertex3f(0.0f, 0.0f, 0.0f);   // 足元
      glVertex3f(0.0f, 0.0f, 0.3f);   // 膝

      // 膝 -> 腰
      glVertex3f(0.0f, 0.0f, 0.3f);
      glVertex3f(0.0f, 0.0f, 0.5f);

      // 腰 -> 胸 (体幹)
      glVertex3f(0.0f, 0.0f, 0.5f);
      glVertex3f(0.0f, 0.0f, 0.7f);

      // 腰付近 -> もう片足 (左右の足に分かれる例)
      glVertex3f(0.0f, 0.0f, 0.3f);
      glVertex3f(0.1f, 0.0f, 0.0f);
      glVertex3f(0.0f, 0.0f, 0.3f);
      glVertex3f(-0.1f, 0.0f, 0.0f);

      // 胸 -> 肩 -> 腕
      glVertex3f(0.0f, 0.0f, 0.7f); 
      glVertex3f(0.2f, 0.0f, 0.6f); // 右腕
      glVertex3f(0.0f, 0.0f, 0.7f);
      glVertex3f(-0.2f, 0.0f, 0.6f); // 左腕

    glEnd();

    // --------------------------------
    // 頭を小さな球体で描画
    // --------------------------------
    glTranslatef(0.0f, 0.0f, 0.75f); // 胸からさらに少し上に
    glutSolidSphere(0.08, 12, 12);

    glPopMatrix();
}


void read_json(const std::string &json_path) {
    std::ifstream file(json_path);
    if (!file.is_open()) {
        std::cerr << "File open error: " << json_path << std::endl;
        return;
    }

    json data;
    try {
        file >> data;
    } catch (const std::exception &e) {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
        return;
    }

    for (auto it = data.begin(); it != data.end(); ++it) {
        int frame = std::stoi(it.key());
        FrameDetections detections;
        for (const auto &item : it.value()) {
            Detection det;
            det.x = item[0];
            det.y = item[1];
            auto color = item[2];
            det.r = color[0];
            det.g = color[1];
            det.b = color[2];
            if (item.size() > 3){
            int isplayer = item[3];
            if (isplayer == 1){
                det.isplayer = true;
            }else{  
                det.isplayer = false;
            }
            }
            if (det.x <0 || det.y < 0 || det.x > 1 || det.y > 1) {
                continue;
            }
            rescaleDetection(det);
            detections.push_back(det);
        }
        all_frames[frame] = detections;
    }

    for (auto &kv : all_frames) {
        sorted_frames.push_back(kv.first);
    }
    std::sort(sorted_frames.begin(), sorted_frames.end());

    return ;
    // ソート後の順序でログ出力
    for (int frame : sorted_frames) {
        const auto &detections = all_frames[frame];
        for (const auto &det : detections) {
            std::cout << "Frame " << frame
                      << ": x=" << det.x
                      << " y=" << det.y
                      << " color=(" << det.r << "," << det.g << "," << det.b
                      << ")\n";
        }
    }
}



void drawSoccerField() {
    glPushMatrix();
    // スケーリング適用
    glScalef(SCALE_FACTOR, SCALE_FACTOR, 1.0f);
    // 平行移動 (0.5, 0.5, 0) を適用
    //glTranslatef(50.0f, 50.0f, 0.0f);

    // フィールド全体の背景を緑色で塗りつぶす
    glColor3f(0.0f, 0.5f, 0.0f); 
    glBegin(GL_QUADS);
        glVertex3f(-52.5f, -34.0f, 0.0f);
        glVertex3f( 52.5f, -34.0f, 0.0f);
        glVertex3f( 52.5f,  34.0f, 0.0f);
        glVertex3f(-52.5f,  34.0f, 0.0f);
    glEnd();

    glColor3f(1.0f, 1.0f, 1.0f);
    glLineWidth(2.0f);

    // フィールドの外周を描画
    glBegin(GL_LINE_LOOP);
        glVertex3f(-52.5f, -34.0f, 0.01f);
        glVertex3f( 52.5f, -34.0f, 0.01f);
        glVertex3f( 52.5f,  34.0f, 0.01f);
        glVertex3f(-52.5f,  34.0f, 0.01f);
    glEnd();

    // センターバー線を描画
    glBegin(GL_LINES);
        glVertex3f(0.0f, -34.0f, 0.01f);
        glVertex3f(0.0f,  34.0f, 0.01f);
    glEnd();

    // センターチャペル（中央円）を描画
    float centerRadius = 9.15f;
    int segments = 64;
    glBegin(GL_LINE_LOOP);
    for(int i = 0; i < segments; ++i) {
        float theta = 2.0f * 3.1415926f * float(i) / float(segments);
        float dx = centerRadius * cosf(theta);
        float dy = centerRadius * sinf(theta);
        glVertex3f(dx, dy, 0.01f);
    }
    glEnd();

    // 左ペナルティエリア
    glBegin(GL_LINE_LOOP);
        glVertex3f(-52.5f,  20.15f, 0.01f);
        glVertex3f(-36.0f,  20.15f, 0.01f);
        glVertex3f(-36.0f, -20.15f, 0.01f);
        glVertex3f(-52.5f, -20.15f, 0.01f);
    glEnd();

    // 右ペナルティエリア
    glBegin(GL_LINE_LOOP);
        glVertex3f(52.5f,  20.15f, 0.01f);
        glVertex3f(36.0f,  20.15f, 0.01f);
        glVertex3f(36.0f, -20.15f, 0.01f);
        glVertex3f(52.5f, -20.15f, 0.01f);
    glEnd();

    // 左ゴールエリア
    glBegin(GL_LINE_LOOP);
        glVertex3f(-52.5f,  8.0f, 0.01f);
        glVertex3f(-44.5f,  8.0f, 0.01f);
        glVertex3f(-44.5f, -8.0f, 0.01f);
        glVertex3f(-52.5f, -8.0f, 0.01f);
    glEnd();

    // 右ゴールエリア
    glBegin(GL_LINE_LOOP);
        glVertex3f(52.5f,  8.0f, 0.01f);
        glVertex3f(44.5f,  8.0f, 0.01f);
        glVertex3f(44.5f, -8.0f, 0.01f);
        glVertex3f(52.5f, -8.0f, 0.01f);
    glEnd();

    glPopMatrix();
}

void drawAxis(){
   int length = 1;
    // x :green y: red z: blue
    glBegin(GL_LINES);
    // x
    glColor3f(0.0, 1.0, 0.0);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(length, 0.0, 0.0);
    // y
    glColor3f(1.0, 0.0, 0.0);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(0.0, length, 0.0);
    // z
    glColor3f(0.0, 0.0, 1.0);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(0.0, 0.0, length);
    glEnd();


}


void renderFrame(int frame_number) {
    auto it = all_frames.find(frame_number);
    if (it != all_frames.end()) {
        const FrameDetections &detections = it->second;

        for (const auto &det : detections) {
            //drawSphereAtDetection(det);
            drawHumanModel(det);
                
        }
    }
}
float camera_angle_x = 90.0f;
float camera_angle_y = 0.0f;
float camera_angle_z = 90.0f;
float radius = 1.0f;
void display() {
    //glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // 白背景
    // 黒背景の場合
   //  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    //空色の背景
    glClearColor(0.5f, 0.5f, 1.0f, 1.0f);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();


    // カメラ位置を、マウスドラッグによる回転で動かす
    float rad_x = camera_angle_x * M_PI / 180.0f;
    float rad_z = camera_angle_z * M_PI / 180.0f;
    
    
    // float camX = radius * cos(rad_y) * sin(rad_x);
    // float camY = radius * sin(rad_y);
    // float camZ = radius * cos(rad_y) * cos(rad_x);

    float camX = radius * cos(rad_x)*cos(rad_z);
    float camY = radius * cos(rad_x)*sin(rad_z);
    float camZ = radius * sin(rad_x);

    float upx = 0.0f;
    float upy = 0.0f;
    float upz = 1.0f;

    gluLookAt(camX, camY, camZ,
              0.0, 0.0, 0.0,
                upx, upy, upz);
   // std::cout<< "camX: " << camX << " camY: " << camY << " camZ: " << camZ << std::endl;
    drawSoccerField();

    drawAxis();

    if (!sorted_frames.empty()) {
        int frame_number = sorted_frames[current_frame_index];
        renderFrame(frame_number);
    }

    glutSwapBuffers();
}

void timerFunc(int value) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    if (!sorted_frames.empty()) {
        current_frame_index = (current_frame_index + 1) % sorted_frames.size();
    }
    glutPostRedisplay();
    glutTimerFunc(33, timerFunc, 0);
}


void reshape(int width, int height) {
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (float)width / (float)height, 0.1, 100.0);
    glMatrixMode(GL_MODELVIEW);
}

void mouseButton(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        if (state == GLUT_DOWN) {
            left_button_down = true;
            last_mouse_x = x;
            last_mouse_y = y;
        } else {
            left_button_down = false;
        }
    }
}

void mouseMotion(int x, int y) {
    if (left_button_down) {
        int dx = x - last_mouse_x;
        int dy = y - last_mouse_y;
        float sensitivity = 0.5f;
        camera_angle_x += dx * sensitivity;
        camera_angle_y += dy * sensitivity;
        if (camera_angle_y > 89.0f) camera_angle_y = 89.0f;
        if (camera_angle_y < -89.0f) camera_angle_y = -89.0f;
        last_mouse_x = x;
        last_mouse_y = y;
        glutPostRedisplay();
    }
}

void keyboardCallback(unsigned char key, int x, int y) {
    float angle_step = 5.0f; // 1回のキー操作で動かす角度(度)

    // switch (key) {
    //     case 'w': // 上方向から見下ろす
    //         camera_angle_y += angle_step;
            
    //         break;
    //     case 's': // 下方向
    //         camera_angle_y -= angle_step;
    //         break;
    //     case 'a': // 左回転
    //         camera_angle_x -= angle_step;
    //         break;
    //     case 'd': // 右回転
    //         camera_angle_x += angle_step;
    //         break;
    //     case 27:  // [ESC]キー
    //         exit(0);
    //         break;
    //     default:
    //         break;
    // }
 switch (key) {
        case 'w': // ピッチ上 (上方向)
            camera_angle_x += angle_step;
            break;
        case 's': // ピッチ下 (下方向)
            camera_angle_x -= angle_step;
            break;
        case 'a': // ヨー左回転
            camera_angle_z -= angle_step;
            break;
        case 'd': // ヨー右回転
            camera_angle_z += angle_step;
            break;
        
        case 'z': //zoom in
            radius -= 0.1f;
            radius = std::max(0.1f, radius);
            std::cout << "radius: " << radius << std::endl;
            break;
        case 'x': //zoom out
            radius += 0.1f;
            std::cout << "radius: " << radius << std::endl;
            break;
        case 27:  // [ESC]キー
            exit(0);
            break;
        default:
            break;
    }
    

    glutPostRedisplay(); // 画面の再描画を要求
}

int main(int argc, char **argv) {
    std::string json_path = "./data/annotation.json";
    read_json(json_path);


    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(800, 600);
    glutCreateWindow("OpenGL Sphere with Mouse Viewpoint Control");

    glEnable(GL_DEPTH_TEST);

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutTimerFunc(33, timerFunc, 0);
    glutMouseFunc(mouseButton);
    glutMotionFunc(mouseMotion);

    glutKeyboardFunc(keyboardCallback);

    glutMainLoop();
    return 0;
}
