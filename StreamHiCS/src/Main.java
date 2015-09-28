import weka.core.Instance;
import moa.streams.generators.RandomRBFGenerator;

public class Main {

	public static void main(String[] args) {
		int numInstances = 1000;
		RandomRBFGenerator stream = new RandomRBFGenerator();
		stream.prepareForUse();

		StreamHiCS streamHiCS = new StreamHiCS(10);

		int numberSamples = 0;
		while (stream.hasMoreInstances() && numberSamples < numInstances) {
			Instance inst = stream.nextInstance();
			System.out.println(inst.value(0));
			streamHiCS.add(inst);
			numberSamples++;
			// System.out.println(streamHiCS.getNumberOfInstances());
		}
	}
}
