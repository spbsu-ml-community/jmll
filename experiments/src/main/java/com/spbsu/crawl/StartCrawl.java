package com.spbsu.crawl;

import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.system.RuntimeUtils;
import com.spbsu.crawl.bl.Mob;
import com.spbsu.crawl.bl.map.CrawlGameSessionMap;
import com.spbsu.crawl.bl.map.TerrainType;
import com.spbsu.crawl.bl.map.mapEvents.MapEvent;
import com.spbsu.crawl.data.GameSession;
import com.spbsu.crawl.data.Hero;

import javax.websocket.DeploymentException;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.List;
import java.util.Optional;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Experts League
 * Created by solar on 23/03/16.
 */
public class StartCrawl {
  public static void main(String[] args) throws IOException, InterruptedException, URISyntaxException, DeploymentException {
    final File socketFile = File.createTempFile("crawl", ".socket");
    final RuntimeUtils.BashProcess bash = new RuntimeUtils.BashProcess("experiments/crawl");
    try {
      //noinspection ResultOfMethodCallIgnored
      socketFile.delete();
      bash.exec("bash ./run_server.sh");
      Thread.sleep(1000);
      final WSEndpoint endpoint = new WSEndpoint(new URI("ws://localhost:8080/socket"));
      final GameProcess gameProcess = new GameProcess(endpoint, new GameSession() {
        private final CrawlGameSessionMap crawlMap = new CrawlGameSessionMap();
        private int x;
        private int y;


        private boolean canMoveTo(final int x, final int y) {
          Optional<TerrainType> terrain = crawlMap.terrainOnCurrentLevel(x, y);
          if (terrain.isPresent()) {
            return terrain.get() != TerrainType.WALL;
          }
          return true;
        }

        private Predicate<Mob.Action> moveableDirection = new Predicate<Mob.Action>() {
          @Override
          public boolean test(Mob.Action action) {
            switch (action) {
              case MOVE_DOWN_LEFT:
                return canMoveTo(x - 1, y - 1);
              case MOVE_DOWN_RIGHT:
                return canMoveTo(x + 1, y - 1);
              case MOVE_DOWN:
                return canMoveTo(x, y - 1);
              case MOVE_LEFT:
                return canMoveTo(x - 1, y);
              case MOVE_RIGHT:
                return canMoveTo(x + 1, y);
              case MOVE_UP:
                return canMoveTo(x, y + 1);
              case MOVE_UP_LEFT:
                return canMoveTo(x - 1, y + 1);
              case MOVE_UP_RIGHT:
                return canMoveTo(x + 1, y + 1);
              default:
                return true;
            }
          }
        };

        @Override
        public Hero.Race race() {
          return Hero.Race.Minotaur;
        }

        @Override
        public Hero.Spec spec() {
          return Hero.Spec.Fighter_Axe;
        }


        final Mob.Action[] moves;
        {
          moves = Stream.of(Mob.Action.values()).filter(action -> action.name().startsWith("MOVE"))
                  .collect(Collectors.toList()).toArray(new Mob.Action[]{});
        }
        final FastRandom rng = new FastRandom();

        @Override
        public Mob.Action tick() {
          List<Mob.Action> avaiableActions = Stream.of(moves).filter(moveableDirection).collect(Collectors.toList());
          return avaiableActions.get(rng.nextInt(avaiableActions.size()));
        }

        @Override
        public void updateMap(MapEvent event) {
          crawlMap.systemMapEvent(event);
        }

        @Override
        public void heroPosition(int x, int y) {
          this.x = x;
          this.y = y;
        }

      });
      gameProcess.run();
    }
    catch (Exception e) {
      e.printStackTrace();
      bash.destroy();
    }
    finally {
      bash.waitFor();
      //noinspection ResultOfMethodCallIgnored
      socketFile.delete();
    }
  }
}
