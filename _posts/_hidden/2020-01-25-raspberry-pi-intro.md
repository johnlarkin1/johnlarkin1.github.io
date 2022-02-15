# 


Helpful links:
* https://www.raspberrypi.org/documentation/configuration/wireless/headless.md
* https://www.raspberrypi.org/documentation/remote-access/ssh/README.md#3-enable-ssh-on-a-headless-raspberry-pi-add-file-to-sd-card-on-another-machine

Try to ssh on using 

ssh pi@raspberrypi.local

or if you have Fing  you should see something like:

<insert image>

Setting up Pi-hole

Definition upstream dns provider -> 
a server that provides service to another server 
located at a higher level in the hierarchy of servers 
in the DNS (domain name system) a name server in a company's LAN (local area netowrk) oftern forwards requests ot the interna service provider's (ISP's) name servers. ISP name servers are upstream to the local server. 



```
Setting up lighttpd (1.4.53-4+deb10u1) ...
Enabling unconfigured: ok
Run "service lighttpd force-reload" to enable changes
Created symlink /etc/systemd/system/multi-user.target.wants/lighttpd.service → /lib/systemd/system/lighttpd.service.
Setting up php7.3-sqlite3 (7.3.19-1~deb10u1) ...
```


```
Configure your devices to use the Pi-hole as their DNS server
using:

IPv4:        192.168.1.165
IPv6:        Not Configured

If you set a new IP address, you should restart the Pi.

The install log is in /etc/pihole.

View the web interface at http://pi.hole/admin or
http://192.168.1.165/admin

Your Admin Webpage login password is neB4yqKR
```