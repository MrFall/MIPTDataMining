#include <iostream>
#include <iomanip>

const int x = 14;
const int y = 14;

using namespace std;

int main() {
	setlocale(0, "Rus");

	int Cell[x][y] = {
		{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1},
		{-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1},
		{-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1},
		{-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1},
		{-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1},
		{-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1},
		{-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1},
		{-1, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, -1},
		{-1, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, -1},
		{-1, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, -1},
		{-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1},
		{-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1},
		{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
	};

	for (int i = 1; i < x - 1; i++)
	{
		for (int j = 1; j < y - 1; j++)
		{
			if (Cell[i][j] == 0)
				cout << "o" << " ";
			if (Cell[i][j] == -1)
				cout << "#" << " ";
		}
		cout << endl;
	}

	int x_A, y_A, x_B, y_B;
	bool flag = 1;
	while (flag) {
		cout << "Ââåäèòå êîîðäèíàòû À:" << endl;
		cin >> x_A >> y_A;
		if ((x_A < 1 || x_A > x - 2) || (y_A < 1 || y_A > y - 2))
			cout << "Âûõîä çà ïðåäåëû ìàññèâà!" << endl;
		else
			flag = false;
	}
	flag = true;
	while (flag) {
		cout << "Ââåäèòå êîîðäèíàòû B:" << endl;
		cin >> x_B >> y_B;
		if ((x_B < 1 || x_B > x - 2) || (y_B < 1 || y_B > y - 2))
			cout << "Âûõîä çà ïðåäåëû ìàññèâà!" << endl;
		else
			flag = false;
	}

	Cell[x_A][y_A] = 255;
	Cell[x_B][y_B] = 256;
	if (y_A != y - 1)
		Cell[x_A][y_A + 1] = 1;
	if (x_A != x - 1)
		Cell[x_A + 1][y_A] = 1;
	if (y_A != 1)
		Cell[x_A][y_A - 1] = 1;
	if (x_A != 1)
		Cell[x_A - 1][y_A] = 1;

	for (int waves = 1; waves < 90; waves++) {
		for (int i = 1; i < x - 1; i++) {
			for (int j = 1; j < y - 1; j++) {
				if (Cell[i][j] == waves) {
					if (Cell[i][j + 1] == 0)
						Cell[i][j + 1] = waves + 1;
					if (Cell[i + 1][j] == 0)
						Cell[i + 1][j] = waves + 1;
					if (Cell[i][j - 1] == 0)
						Cell[i][j - 1] = waves + 1;
					if (Cell[i - 1][j] == 0)
						Cell[i - 1][j] = waves + 1;
				}
			}
		}
	}

	
	for (int i = 1; i < x - 1; i++) {
		for (int j = 1; j < y - 1; j++) {
			cout << setw(4) << Cell[i][j];
		}
		cout << endl;
	}

	system("pause");
	return 0;
}
