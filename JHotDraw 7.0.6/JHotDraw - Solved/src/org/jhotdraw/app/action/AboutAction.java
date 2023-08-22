/*
 * @(#)AboutAction.java  1.0  04 January 2005
 *
 * Copyright (c) 1996-2006 by the original authors of JHotDraw
 * and all its contributors ("JHotDraw.org")
 * All rights reserved.
 *
 * This software is the confidential and proprietary information of
 * JHotDraw.org ("Confidential Information"). You shall not disclose
 * such Confidential Information and shall use it only in accordance
 * with the terms of the license agreement you entered into with
 * JHotDraw.org.
 */

package org.jhotdraw.app.action;

import org.jhotdraw.util.*;

import java.awt.*;
import java.awt.event.*;
import java.io.IOException;
import java.net.URI;
import javax.swing.*;
import javax.swing.event.HyperlinkEvent;
import javax.swing.event.HyperlinkListener;

import org.jhotdraw.app.*;

/**
 * Displays a dialog showing information about the application.
 *
 * @author  Werner Randelshofer
 * @version 1.0  04 January 2005  Created.
 */
public class AboutAction extends AbstractApplicationAction {
    public final static String ID = "about";
    
    /** Creates a new instance. */
    public AboutAction(Application app) {
        super(app);
        ResourceBundleUtil labels = ResourceBundleUtil.getLAFBundle("org.jhotdraw.app.Labels");
        labels.configureAction(this, ID);
        }
    
    public void actionPerformed(ActionEvent evt) {
        Application app = getApplication();
        // for copying style
        JLabel label = new JLabel();
        Font font = label.getFont();

        // create some css from the label's font
        StringBuffer style = new StringBuffer("font-family:" + font.getFamily() + ";");
        style.append("font-weight:" + (font.isBold() ? "bold" : "normal") + ";");
        style.append("font-size:" + font.getSize() + "pt;");

        // html content
        JEditorPane ep = new JEditorPane("text/html", "<html><body style=\"" + style + "\">" //
                + app.getName()+" "+app.getVersion()+"<br>"+app.getCopyright()+
                "<br><br>Running on Java "+System.getProperty("java.vm.version")+
                ", "+System.getProperty("java.vendor")
                + "<br> Some text, and <a href=\"http://google.com/\">a link</a>" //
                + "</body></html>");

        // handle link events
        ep.addHyperlinkListener(new HyperlinkListener()
        {
            @Override
            public void hyperlinkUpdate(HyperlinkEvent e)
            {
                if (e.getEventType().equals(HyperlinkEvent.EventType.ACTIVATED)) {
                    try {
                        Desktop.getDesktop().browse(URI.create(e.getURL().toString())); // roll your own link launcher or use Desktop if J6+
                    } catch (IOException ex) {
                        throw new RuntimeException(ex);
                    }
                }
            }
        });
        ep.setEditable(false);
        ep.setBackground(label.getBackground());


        JOptionPane.showMessageDialog(app.getComponent(),
                ep,
                "About", JOptionPane.PLAIN_MESSAGE);
    }
}
