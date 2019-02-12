#include "glaux.h"
#include "glut.h"

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glut32.lib")
#pragma comment(lib, "glaux.lib")

AUX_RGBImageRec  *p_texture = NULL;
AUX_RGBImageRec  *p_texture_1 = NULL;
unsigned int      texture_id = 0;
unsigned int      texture_id_1 = 0;
float x=0,y=0,z=0,x_c=0,y_c=0, size = -5;
int flag=1;

void Read_kb(unsigned char key,int ,int){
	if(key=='o') z+=10;
	if(key=='p') z-=10;
	if(key=='k') y+=10;
	if(key=='l') y-=10;
	if(key=='n') x+=10;
	if(key=='m') x-=10;

	if(key==27) exit(1);
	if(key=='w') y_c+=0.1;
	if(key=='s') y_c-=0.1;
	if(key=='a') x_c-=0.1;
	if(key=='d') x_c+=0.1;
	if(key=='t') size += 1;
	if(key=='y') size -= 1;
	if (key==32) flag*=(-1);

	glutPostRedisplay();
}

void RenderScene() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glViewport(0, 0, 600, 600);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(45.0f, 1.0f, 0.1f, 1000.0f);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  
  glTranslatef(x_c,y_c, size);
  glRotatef(x,0,0,1);
  glRotatef(y,0,1,0);
  glRotatef(z,1,0,0);

  glBindTexture(GL_TEXTURE_2D, texture_id_1);
  glBegin(GL_QUADS);
  glColor3f(0.5,0.5,0.2);
  glTexCoord2f(0, 0);
  glVertex3f(-0.5, -0.5, 0.5);
  glTexCoord2f(1, 0);
  glVertex3f(0.5, -0.5, 0.5);
  glTexCoord2f(1, 1);
  glVertex3f(0.5, 0.5, 0.5);
  glTexCoord2f(1, 0);
  glVertex3f(-0.5, 0.5, 0.5);
  glEnd();

  glBindTexture(GL_TEXTURE_2D, texture_id);
  glBegin(GL_QUADS);
  //glColor3f(0.5,0.1,0.2);
  glTexCoord2f(0, 0);
  glVertex3f(-0.5,-0.5,-0.5);
  glTexCoord2f(1, 0);
  glVertex3f(0.5,-0.5,-0.5);
  glTexCoord2f(1, 1);
  glVertex3f(0.5,0.5,-0.5);
  glTexCoord2f(0, 1);
  glVertex3f(-0.5,0.5,-0.5);
  glEnd();
 
  glutSwapBuffers();
}

int main(int argc, char *argv[]) {
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutInitWindowPosition(100, 100);
  glutInitWindowSize(600, 600);
  glutCreateWindow("Пример вывода 3D графики");
  glEnable(GL_DEPTH_TEST);
  glClearColor(0.5, 0.5, 0.5, 1);

  p_texture = auxDIBImageLoad("data/grass_01.bmp");
  if (!p_texture)
    return 1;
  glGenTextures(1, &texture_id);
  glBindTexture(GL_TEXTURE_2D, texture_id);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, 3, p_texture->sizeX, p_texture->sizeY, 0, GL_RGB, GL_UNSIGNED_BYTE, p_texture->data);
  if (p_texture) {
    if (p_texture->data)
      free(p_texture->data);
    free(p_texture);
  }

  p_texture_1 = auxDIBImageLoad("data/grass_02.bmp");
  if (!p_texture_1)
    return 1;
  glGenTextures(1, &texture_id_1);
  glBindTexture(GL_TEXTURE_2D, texture_id_1);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, 3, p_texture_1->sizeX, p_texture_1->sizeY, 0, GL_RGB, GL_UNSIGNED_BYTE, p_texture_1->data);
  if (p_texture_1) {
    if (p_texture_1->data)
      free(p_texture_1->data);
    free(p_texture_1);
  }


  glEnable(GL_TEXTURE_2D);

  glutDisplayFunc(RenderScene);
  glutKeyboardFunc(Read_kb);

  glutMainLoop();

  return 0;
}