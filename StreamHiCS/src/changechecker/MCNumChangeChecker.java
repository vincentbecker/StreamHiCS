package changechecker;

import contrast.Contrast;

/**
 * Signals a change when the number of microclusters changes significantly (using the Page-Hinkley Test).
 * 
 * @author Vincent
 *
 */
public class MCNumChangeChecker extends ChangeChecker{

	private Contrast contrastEvaluator;
	private double magnitudeThreshold;
	private double detectionThreshold;
	private double increasingMT = Double.MAX_VALUE;
	private double decreasingMT = Double.MIN_VALUE;
	
	public MCNumChangeChecker(int checkInterval, Contrast contrastEvaluator, double magnitudeThreshold, double detectionThreshold) {
		super(checkInterval);
		this.contrastEvaluator = contrastEvaluator;
		this.magnitudeThreshold = magnitudeThreshold;
		this.detectionThreshold = detectionThreshold;
	}
	
	@Override
	public boolean checkForChange() {
		double mt = contrastEvaluator.getNumberOfElements() - magnitudeThreshold;

		increasingMT = Math.min(increasingMT, mt);
		decreasingMT = Math.max(decreasingMT, mt);

		double increasingPHT = mt - increasingMT;
		double decreasingPHT = decreasingMT - mt;
		System.out.println("IncreasingPHT: " + increasingPHT + ", DecreasingPHT: " + decreasingPHT);
		if (increasingPHT > detectionThreshold || decreasingPHT > detectionThreshold) {
			reset();
			return true;
		}
		
		return false;
	}

	private void reset() {
		increasingMT = Double.MAX_VALUE;
		decreasingMT = Double.MIN_VALUE;
	}
}
