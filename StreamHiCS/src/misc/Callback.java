package misc;
/**
 * This class represents a callback.
 * 
 * @author Vincent
 *
 */
public interface Callback {

	/**
	 * If a change if noted, the implementing class can be notified by calling
	 * this method.
	 */
	public void onAlarm();
}
