import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Comparator;
import java.util.List;

import moa.streams.generators.WaveformGenerator;
import moa.tasks.WriteStreamToARFFFile;

public class Main {

	public enum Dataset {
		COVERTYPE, COVERTYPE_FILTERED, INTRUSION_DETECTION, INTRUSION_DETECTION_FILTERED, ELECTRICITY, WAVEFORM
	};

	public static void main(String[] args) {

		Dataset set = Dataset.WAVEFORM;

		String headerPath = null;
		String inputPath = null;
		String outputPath = null;
		Comparator<String> comparator = null;
		int[] filterColumns = null;
		switch (set) {
		case COVERTYPE:
			headerPath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Implementation/StreamHiCS/StreamHiCS/Tests/RealWorldData/covertypeHeader.txt";
			inputPath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Implementation/StreamHiCS/StreamHiCS/Tests/RealWorldData/covertypeNorm.txt";
			outputPath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Implementation/StreamHiCS/StreamHiCS/Tests/RealWorldData/covertypeNorm_sorted.arff";
			comparator = new CovertypeComparator();
			break;
		case COVERTYPE_FILTERED:
			headerPath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Implementation/StreamHiCS/StreamHiCS/Tests/RealWorldData/covertypeHeader_filtered.txt";
			inputPath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Implementation/StreamHiCS/StreamHiCS/Tests/RealWorldData/covertypeNorm.txt";
			outputPath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Implementation/StreamHiCS/StreamHiCS/Tests/RealWorldData/covertypeNorm_sorted_filtered.arff";
			comparator = new CovertypeComparator();
			filterColumns = new int[44];
			for (int i = 0; i < 44; i++) {
				filterColumns[i] = i + 10;
			}
			break;
		case INTRUSION_DETECTION:
			headerPath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Implementation/StreamHiCS/StreamHiCS/Tests/RealWorldData/kddHeader.txt";
			inputPath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Implementation/StreamHiCS/StreamHiCS/Tests/RealWorldData/kddcup99_10_percent.txt";
			outputPath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Implementation/StreamHiCS/StreamHiCS/Tests/RealWorldData/kddcup99_10_percent_sorted.arff";
			// inputPath = "D:/Informatik/MSc/IV/Masterarbeit
			// Porto/Implementation/StreamHiCS/StreamHiCS/Tests/RealWorldData/kddcup99.txt";
			// outputPath = "D:/Informatik/MSc/IV/Masterarbeit
			// Porto/Implementation/StreamHiCS/StreamHiCS/Tests/RealWorldData/kddcup99_sorted.txt";
			comparator = new IntrusionDetectionComparator();
			filterColumns = new int[8];
			filterColumns[0] = 0;
			filterColumns[1] = 1;
			filterColumns[2] = 2;
			filterColumns[3] = 3;
			filterColumns[4] = 6;
			filterColumns[5] = 11;
			filterColumns[6] = 20;
			filterColumns[7] = 21;
			break;
		case INTRUSION_DETECTION_FILTERED:
			headerPath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Implementation/StreamHiCS/StreamHiCS/Tests/RealWorldData/kddHeader_filtered.txt";
			inputPath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Implementation/StreamHiCS/StreamHiCS/Tests/RealWorldData/kddcup99_10_percent.txt";
			outputPath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Implementation/StreamHiCS/StreamHiCS/Tests/RealWorldData/kddcup99_10_percent_sorted_filtered.arff";
			// inputPath = "D:/Informatik/MSc/IV/Masterarbeit
			// Porto/Implementation/StreamHiCS/StreamHiCS/Tests/RealWorldData/kddcup99.txt";
			// outputPath = "D:/Informatik/MSc/IV/Masterarbeit
			// Porto/Implementation/StreamHiCS/StreamHiCS/Tests/RealWorldData/kddcup99_sorted.txt";
			comparator = new IntrusionDetectionComparator();
			filterColumns = new int[18];
			filterColumns[0] = 1;
			filterColumns[1] = 2;
			filterColumns[2] = 3;
			filterColumns[3] = 6;
			filterColumns[4] = 7;
			filterColumns[5] = 8;
			filterColumns[6] = 10;
			for (int i = 7; i < 18; i++) {
				filterColumns[i] = i + 4;
			}
			break;
		case ELECTRICITY:
			inputPath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Implementation/StreamHiCS/StreamHiCS/Tests/RealWorldData/elecNormNew.txt";
			outputPath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Implementation/StreamHiCS/StreamHiCS/Tests/RealWorldData/elecNormNew_sorted.txt";
			comparator = new ElectricityComparator();
			break;
		default:
			break;
		case WAVEFORM:
			headerPath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Implementation/StreamHiCS/StreamHiCS/Tests/waveformHeader.txt";
			inputPath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Implementation/StreamHiCS/StreamHiCS/Tests/waveform.arff";
			outputPath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Implementation/StreamHiCS/StreamHiCS/Tests/waveform_sorted.arff";
			WriteStreamToARFFFile writer = new WriteStreamToARFFFile();
			WaveformGenerator stream = new WaveformGenerator();
			stream.addNoiseOption.set();
			stream.prepareForUse();
			writer.streamOption.setCurrentObject(stream);
			writer.arffFileOption.setValue(inputPath);
			writer.maxInstancesOption.setValue(100000);
			writer.suppressHeaderOption.setValue(true);
			writer.doTask();
			
			comparator = new WaveformComparator();
		}

		sortAndWrite(headerPath, inputPath, outputPath, comparator, filterColumns);
	}

	private static void sortAndWrite(String headerPath, String inputPath, String outputPath,
			Comparator<String> comparator, int[] filterColumns) {
		try {
			// Header
			List<String> header = Files.readAllLines(Paths.get(headerPath), StandardCharsets.UTF_8);
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
					// String classLabel = splitLine[splitLine.length - 1];
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
