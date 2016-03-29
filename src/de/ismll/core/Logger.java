package de.ismll.core;

import java.util.logging.ConsoleHandler;
import java.util.logging.Handler;
import java.util.logging.Level;

public class Logger
{
	public static final Level LEVEL = Level.INFO;
	
	private static java.util.logging.Logger log = Logger.getInstance();

	private static java.util.logging.Logger getInstance()
	{
		java.util.logging.Logger log = java.util.logging.Logger.getAnonymousLogger();
		log.setUseParentHandlers(false);
		Handler h = new ConsoleHandler();
		h.setLevel(LEVEL);
		log.addHandler(h);
		log.setLevel(LEVEL);
		return log;
	}

	public static void warning(String msg)
	{
		log.warning(msg);
	}

	public static void fine(String msg)
	{
		log.fine(msg);
	}

	public static void finer(String msg)
	{
		log.finer(msg);
	}

	public static void finest(String msg)
	{
		log.finest(msg);
	}

	public static void info(String msg)
	{
		log.info(msg);
	}
	
	public static void severe(String msg)
	{
		log.severe(msg);
	}
}
