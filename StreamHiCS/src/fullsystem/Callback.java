package fullsystem;

/**
 * This class represents a callback.
 * 
 * @author Vincent
 *
 */
public interface Callback {

	/**
	 * If a change is detected, the implementing class can be notified by
	 * calling this method.
	 */
	public void onAlarm();
}
