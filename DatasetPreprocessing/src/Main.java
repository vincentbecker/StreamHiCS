import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Comparator;
import java.util.List;

public class Main {

	public enum Dataset {
		COVERTYPE, INTRUSION_DETECTION, ELECTRICITY
	};

	public static void main(String[] args) {

		Dataset set = Dataset.INTRUSION_DETECTION;

		String inputPath = null;
		String outputPath = null;
		Comparator<String> comparator = null;
		int[] filterColumns = null;
		switch (set) {
		case COVERTYPE:
			inputPath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Implementation/StreamHiCS/StreamHiCS/Tests/RealWorldData/Covertype.txt";
			outputPath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Implementation/StreamHiCS/StreamHiCS/Tests/RealWorldData/Covertype_sorted.txt";
			comparator = new CovertypeComparator();
			break;
		case INTRUSION_DETECTION:
			//inputPath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Implementation/StreamHiCS/StreamHiCS/Tests/RealWorldData/kddcup99_10_percent.txt";
			//outputPath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Implementation/StreamHiCS/StreamHiCS/Tests/RealWorldData/kddcup99_10_percent_sorted.txt";
			inputPath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Implementation/StreamHiCS/StreamHiCS/Tests/RealWorldData/kddcup99.txt";
			outputPath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Implementation/StreamHiCS/StreamHiCS/Tests/RealWorldData/kddcup99_sorted.txt";
			comparator = new IntrusionDetectionComparator();
			filterColumns = new int[7];
			filterColumns[0] = 1;
			filterColumns[1] = 2;
			filterColumns[2] = 3;
			filterColumns[3] = 6;
			filterColumns[4] = 11;
			filterColumns[5] = 20;
			filterColumns[6] = 21;
			break;
		case ELECTRICITY:
			inputPath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Implementation/StreamHiCS/StreamHiCS/Tests/RealWorldData/elecNormNew.txt";
			outputPath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Implementation/StreamHiCS/StreamHiCS/Tests/RealWorldData/elecNormNew_sorted.txt";
			comparator = new ElectricityComparator();
			break;
		default:
			break;
		}

		sortAndWrite(inputPath, outputPath, comparator, filterColumns);
	}

	private static void sortAndWrite(String inputPath, String outputPath, Comparator<String> comparator,
			int[] filterColumns) {
		try {
			//Header
			List<String> header = Files.readAllLines(Paths.get("D:/Informatik/MSc/IV/Masterarbeit Porto/Implementation/StreamHiCS/StreamHiCS/Tests/RealWorldData/kddARFFHeader.txt"), StandardCharsets.UTF_8);
			Files.write(Paths.get(outputPath), header);
			
			List<String> lines = Files.readAllLines(Paths.get(inputPath), StandardCharsets.UTF_8);
			if (filterColumns != null) {
				for (int i = 0; i < lines.size(); i++) {
					String line = lines.get(i);
					String[] splitLine = line.split(",");
					for (int j = 0; j < filterColumns.length; j++) {
						splitLine[filterColumns[j]] = null;
					}
					String newLine = "";
					for (int j = 0; j < splitLine.length - 1; j++) {
						if (splitLine[j] != null) {
							newLine += splitLine[j] + ",";
						}
					}
					String classLabel = splitLine[splitLine.length - 1].replace(".", "");
					newLine += classLabel;
					lines.set(i, newLine);
				}
			}
			lines.sort(comparator);
			Files.write(Paths.get(outputPath), lines, StandardOpenOption.APPEND);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
