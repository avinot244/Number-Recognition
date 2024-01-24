package main_package;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.RenderingHints;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.swing.JComponent;
/**
 * 
 * @author Aymeric2
 *
 */
public class DrawArea extends JComponent{
	//Image where we're going to draw stuff
	private Image image;
	private Graphics2D g2;
	
	private int currentX, currentY, oldX, oldY;
	
	public Image getImage() {
		return this.image;
	}
	
	public void savePic(Image image, String type, String dst) {
		BufferedImage resizedImage = new BufferedImage(100, 100, BufferedImage.TYPE_3BYTE_BGR);
		Graphics2D graphics2D = resizedImage.createGraphics();
		graphics2D.drawImage(image, 0, 0, 100, 100, null);
		graphics2D.dispose();
		BufferedImage bi = (BufferedImage) resizedImage;
		try {
			ImageIO.write(bi, type, new File(dst));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public DrawArea (){
		setDoubleBuffered(false);
		addMouseListener(new MouseAdapter() {
			public void mousePressed(MouseEvent e) {
				//Save coord x, y when mouse is pressed
				oldX = e.getX();
				oldY = e.getY();
			}
		});
		
		addMouseMotionListener(new MouseMotionAdapter() {
			public void mouseDragged(MouseEvent e) {
				//coord x, y when drag mouse
				currentX = e.getX();
				currentY = e.getY();
				
				if (g2 != null) {
					// we draw a line if g2 context is not null
					g2.setStroke(new BasicStroke(5));
					g2.drawLine(oldX, oldY, currentX, currentY);
					//refresh draw area and coordinates of the mouse
					repaint();
					oldX = currentX;
					oldY = currentY;
				}
			}
		});
	}
	protected void paintComponent(Graphics g) {
		if (image == null) {
			//image to draw is null -> create a new one
			image = createImage(getSize().width, getSize().height); //TODO : set size to 100x100
			g2 = (Graphics2D) image.getGraphics();
			//enable antialiasing
			g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, 
					RenderingHints.VALUE_ANTIALIAS_ON);
			//clear draw area
			clear();
		}
		g.drawImage(image, 0, 0, null);
	}
	
	//creating exposed method
	
	public void clear() {
		g2.setPaint(Color.white);
		//filling draw area of white to clear it
		g2.fillRect(0, 0, getSize().width, getSize().height);
		g2.setPaint(Color.black);
		repaint();
	}
	
	public void black() {
		g2.setPaint(Color.black);
	}
		
}
