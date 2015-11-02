import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class CSVReader {

	public double[][] read(String fileName) {

		BufferedReader br = null;
		String line = "";
		String separator = ",";

		ArrayList<String[]> temp = new ArrayList<String[]>();

		try {
			br = new BufferedReader(new FileReader(new File(fileName)));
			while ((line = br.readLine()) != null) {
				temp.add(line.split(separator));
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (br != null) {
				try {
					br.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}

		int m = temp.size();
		int n = temp.get(0).length;
		double[][] result = new double[m][n];
		String[] lineContent;
		for (int i = 0; i < m; i++) {
			lineContent = temp.get(i);
			for (int j = 0; j < n; j++) {
				result[i][j] = Double.parseDouble(lineContent[j]);
			}
		}

		return result;
	}

}
